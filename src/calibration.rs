//! Guided calibration: fits the raw `(o1, o2, o3)` -> `(pitch, yaw, roll)`
//! map from continuous head motion.
//!
//! For each head axis the user oscillates around that axis only. The samples
//! (raw channel deltas from the neutral pose) are linearly detrended to drop
//! slow drift, and the principal component of what remains is the raw-space
//! direction that axis moves along. The three directions form the columns of
//! the mixing matrix `D`; its inverse decouples the axes. Each row is scaled
//! so the output keeps the LibrePods degree convention, and each direction's
//! sign is chosen to agree with the uncalibrated formulas so axis signs do not
//! flip after calibrating.

use crate::head_tracking::{Calibration, DEG_PER_RAW};
use anyhow::{anyhow, bail, Context, Result};
use log::{info, warn};
use std::path::Path;

/// One calibration sample: seconds since phase start, raw deltas from neutral.
pub type Sample = (f64, [f64; 3]);

/// Fewer samples than this (~1 s of stream) cannot describe a motion.
const MIN_SAMPLES: usize = 20;
/// Minimum std along the motion direction, in raw units (~3 deg).
const MIN_STD: f64 = 500.0;
/// Below this share of variance on the main direction the motion was mixed.
const MIN_PURITY: f64 = 0.6;
const WARN_PURITY: f64 = 0.85;
/// Unit motion directions must span a real volume (1.0 = orthogonal).
const MIN_DETERMINANT: f64 = 0.3;

/// Principal motion direction of one calibration phase.
#[derive(Debug, Clone, Copy)]
pub struct AxisFit {
    /// Unit direction in raw `(o1, o2, o3)` space (sign arbitrary).
    pub dir: [f64; 3],
    /// Standard deviation along `dir`, raw units.
    pub std: f64,
    /// Share of the phase's variance explained by `dir` (0..1).
    pub purity: f64,
}

/// Principal motion direction of one phase, after removing linear drift.
pub fn fit_axis(samples: &[Sample]) -> Result<AxisFit> {
    if samples.len() < MIN_SAMPLES {
        bail!(
            "only {} head-tracking samples captured; is the AirPods stream active?",
            samples.len()
        );
    }
    let n = samples.len() as f64;
    let t_mean = samples.iter().map(|s| s.0).sum::<f64>() / n;
    let t_var = samples.iter().map(|s| (s.0 - t_mean).powi(2)).sum::<f64>();

    let mut residuals = vec![[0.0f64; 3]; samples.len()];
    for c in 0..3 {
        let x_mean = samples.iter().map(|s| s.1[c]).sum::<f64>() / n;
        let slope = if t_var > 0.0 {
            samples
                .iter()
                .map(|s| (s.0 - t_mean) * (s.1[c] - x_mean))
                .sum::<f64>()
                / t_var
        } else {
            0.0
        };
        for (r, s) in residuals.iter_mut().zip(samples) {
            r[c] = s.1[c] - x_mean - slope * (s.0 - t_mean);
        }
    }

    let mut cov = [[0.0f64; 3]; 3];
    for r in &residuals {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += r[i] * r[j] / n;
            }
        }
    }
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    if trace <= 0.0 {
        bail!("no motion captured");
    }

    // Power iteration from the column with the largest variance.
    let start = (0..3)
        .max_by(|&a, &b| cov[a][a].total_cmp(&cov[b][b]))
        .unwrap_or(0);
    let mut v = normalize([cov[0][start], cov[1][start], cov[2][start]]);
    for _ in 0..200 {
        v = normalize(mat_vec(&cov, v));
    }
    let lambda = dot(v, mat_vec(&cov, v)).max(0.0);

    Ok(AxisFit {
        dir: v,
        std: lambda.sqrt(),
        purity: lambda / trace,
    })
}

/// Build the decoupling matrix from the three fitted motion directions.
pub fn build_calibration(pitch: AxisFit, yaw: AxisFit, roll: AxisFit) -> Result<Calibration> {
    let axes = [("pitch", pitch), ("yaw", yaw), ("roll", roll)];
    for (name, fit) in axes {
        if fit.std < MIN_STD {
            bail!(
                "{name} motion too small (~{:.1} deg); move further",
                fit.std * DEG_PER_RAW as f64
            );
        }
        if fit.purity < MIN_PURITY {
            bail!(
                "{name} motion was mixed with other axes ({:.0}% clean); move only that axis",
                fit.purity * 100.0
            );
        }
        if fit.purity < WARN_PURITY {
            warn!(
                "{name} motion only {:.0}% clean; recalibrate if axes still couple",
                fit.purity * 100.0
            );
        }
    }

    // Sign references: the uncalibrated formulas (LibrePods pitch/yaw and
    // the AirPods 4 roll fit from pods-head-tracker).
    let references = [
        [0.0, 1.0, 1.0],   // pitch = (o2 + o3) / 2
        [0.0, 1.0, -1.0],  // yaw   = (o2 - o3) / 2
        [1.0, -0.5, -0.5], // roll  = (o1 - (o2 + o3) / 2) / 2
    ];
    let mut dirs = [pitch.dir, yaw.dir, roll.dir];
    for ((dir, reference), (name, _)) in dirs.iter_mut().zip(references).zip(axes) {
        let agreement = dot(*dir, normalize(reference));
        if agreement < 0.0 {
            *dir = dir.map(|x| -x);
        }
        if agreement.abs() < 0.3 {
            warn!("{name} direction is far from the default formula; its sign may be inverted (fix in opentrack Mapping if so)");
        }
    }

    // D has the motion directions as columns.
    let d = [
        [dirs[0][0], dirs[1][0], dirs[2][0]],
        [dirs[0][1], dirs[1][1], dirs[2][1]],
        [dirs[0][2], dirs[1][2], dirs[2][2]],
    ];
    let det = det3(&d);
    if det.abs() < MIN_DETERMINANT {
        bail!(
            "the three motions overlap too much (independence {:.2}); redo each axis separately",
            det.abs()
        );
    }
    let inv = inverse3(&d, det);

    // Gain: degrees per raw unit along each direction, matching the
    // LibrePods convention (its pitch/yaw see the (o2, o3) projection / sqrt2).
    let mut m = [[0.0f32; 3]; 3];
    for (axis, dir) in dirs.iter().enumerate() {
        let gain = dir[1].hypot(dir[2]) / std::f64::consts::SQRT_2 * DEG_PER_RAW as f64;
        for c in 0..3 {
            m[axis][c] = (gain * inv[axis][c]) as f32;
        }
    }
    Ok(Calibration { m })
}

/// RMS of each output axis while replaying one phase's (detrended) samples.
/// For a clean phase only its own axis should be large.
pub fn response_rms(cal: &Calibration, samples: &[Sample]) -> [f64; 3] {
    let n = samples.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| samples.iter().map(|s| s.1[c]).sum::<f64>() / n);
    let mut acc = [0.0f64; 3];
    for s in samples {
        let d = std::array::from_fn(|c| (s.1[c] - mean[c]) as f32);
        let out = cal.apply(d);
        for i in 0..3 {
            acc[i] += (out[i] as f64).powi(2);
        }
    }
    acc.map(|a| (a / n).sqrt())
}

/// Save a calibration matrix.
pub fn save(path: &Path, cal: &Calibration) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = String::from(
        "# airpods-head-track calibration: raw (o1,o2,o3) deltas -> (pitch,yaw,roll) degrees\n",
    );
    for (r, row) in cal.m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            text.push_str(&format!("m{r}{c}={v}\n"));
        }
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Load a calibration matrix. `Ok(None)` if the file does not exist.
pub fn load(path: &Path) -> Result<Option<Calibration>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mut vals = [[None; 3]; 3];
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().as_bytes();
        if key.len() == 3 && key[0] == b'm' {
            let (r, c) = ((key[1] - b'0') as usize, (key[2] - b'0') as usize);
            if r < 3 && c < 3 {
                vals[r][c] = Some(value.trim().parse::<f32>()?);
            }
        }
    }
    let mut m = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = vals[r][c].ok_or_else(|| {
                anyhow!("calibration file is incomplete or from an older version")
            })?;
        }
    }
    Ok(Some(Calibration { m }))
}

/// Log the fit quality of a finished calibration.
pub fn report(cal: &Calibration, phases: [(&str, &[Sample]); 3]) {
    info!("Calibration matrix (rows pitch/yaw/roll):");
    for row in &cal.m {
        info!("  [{:+.6} {:+.6} {:+.6}]", row[0], row[1], row[2]);
    }
    for (axis, (name, samples)) in phases.into_iter().enumerate() {
        let rms = response_rms(cal, samples);
        let own = rms[axis].max(1e-9);
        let leaks: Vec<String> = ["pitch", "yaw", "roll"]
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(i, other)| format!("{other} {:.0}%", rms[i] / own * 100.0))
            .collect();
        info!(
            "  {name}: ~{:.1} deg rms; leak into {}",
            rms[axis],
            leaks.join(", ")
        );
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = dot(v, v).sqrt();
    if len > 0.0 {
        v.map(|x| x / len)
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| dot(m[i], v))
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse3(m: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let c =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 2, 1, 2) / det,
            -c(0, 2, 1, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 2, 0, 2) / det,
            c(0, 2, 0, 2) / det,
            -c(0, 1, 0, 2) / det,
        ],
        [
            c(1, 2, 0, 1) / det,
            -c(0, 2, 0, 1) / det,
            c(0, 1, 0, 1) / det,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw-space response per degree of each head axis for a sensor following
    /// the AirPods 4 model (o1 = pitch + roll, (o2+o3)/2 = pitch - roll,
    /// (o2-o3)/2 = yaw), then rotated by `mount` to mimic a skewed IMU.
    fn mixing(mount: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let k = 1.0 / DEG_PER_RAW as f64;
        let base = [
            [k, k, k],    // pitch
            [0.0, k, -k], // yaw
            [k, -k, -k],  // roll
        ];
        base.map(|v| mat_vec(&mount, v))
    }

    /// Oscillate one head axis (degrees) with drift and cross-axis wobble.
    fn phase(mix: &[[f64; 3]; 3], axis: usize) -> Vec<Sample> {
        (0..200)
            .map(|i| {
                let t = i as f64 * 0.04;
                let mut head = [0.0; 3];
                head[axis] = 25.0 * (t * 3.0).sin();
                head[(axis + 1) % 3] += 1.5 * (t * 7.3).sin(); // involuntary wobble
                head[(axis + 2) % 3] += 0.4 * t; // slow drift
                let raw = std::array::from_fn(|c| (0..3).map(|a| head[a] * mix[a][c]).sum::<f64>());
                (t, raw)
            })
            .collect()
    }

    fn rot_x(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    }

    fn rot_z(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    }

    fn calibrate(mix: &[[f64; 3]; 3]) -> Result<Calibration> {
        let fits: Vec<AxisFit> = (0..3).map(|a| fit_axis(&phase(mix, a)).unwrap()).collect();
        build_calibration(fits[0], fits[1], fits[2])
    }

    #[test]
    fn skewed_mount_is_decoupled_with_correct_signs() {
        let mix = mixing(matmul(rot_x(25.0), rot_z(15.0)));
        let cal = calibrate(&mix).unwrap();
        for axis in 0..3 {
            let raw = mix[axis].map(|x| (x * 10.0) as f32); // +10 deg on one axis
            let out = cal.apply(raw);
            assert!(out[axis] > 5.0, "axis {axis} lost or flipped: {out:?}");
            for other in (0..3).filter(|&o| o != axis) {
                assert!(
                    out[other].abs() < 0.05 * out[axis],
                    "axis {axis} leaks into {other}: {out:?}"
                );
            }
        }
    }

    #[test]
    fn in_plane_twist_keeps_degree_scale() {
        // A twist within the (o2, o3) plane must not change the degree gain.
        let mix = mixing(rot_x(30.0));
        let cal = calibrate(&mix).unwrap();
        for axis in 0..3 {
            let out = cal.apply(mix[axis].map(|x| (x * 10.0) as f32));
            assert!((out[axis] - 10.0).abs() < 0.5, "axis {axis}: {out:?}");
        }
    }

    #[test]
    fn rejects_overlapping_motions() {
        let mix = mixing(rot_x(0.0));
        let pitch = fit_axis(&phase(&mix, 0)).unwrap();
        let yaw = fit_axis(&phase(&mix, 1)).unwrap();
        // "Roll" phase that actually repeated the pitch motion.
        assert!(build_calibration(pitch, yaw, pitch).is_err());
    }

    #[test]
    fn rejects_motion_that_is_too_small() {
        let mix = mixing(rot_x(0.0));
        let still: Vec<Sample> = phase(&mix, 0)
            .into_iter()
            .map(|(t, raw)| (t, raw.map(|x| x * 0.01)))
            .collect();
        let pitch = fit_axis(&still).unwrap();
        let yaw = fit_axis(&phase(&mix, 1)).unwrap();
        let roll = fit_axis(&phase(&mix, 2)).unwrap();
        assert!(build_calibration(pitch, yaw, roll).is_err());
    }

    #[test]
    fn old_two_by_two_file_is_rejected() {
        let path = std::env::temp_dir().join("aht_cal_old_format_test.txt");
        std::fs::write(&path, "m00=1\nm01=0\nm10=0\nm11=1\n").unwrap();
        let result = load(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn save_then_load_round_trips() {
        let path = std::env::temp_dir().join("aht_cal_round_trip_test.txt");
        let cal = Calibration {
            m: [[0.1, -0.2, 0.3], [0.004, 0.5, -0.6], [7.0, 0.08, -0.009]],
        };
        save(&path, &cal).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded, Some(cal));
    }
}
