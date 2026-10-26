//! Head tracking data parser and orientation calculator.
//!
//! The AirPods stream three raw orientation channels (int16 at packet
//! offsets 43/45/47). They are taken relative to a neutral pose and mapped to
//! pitch/yaw/roll either by the LibrePods formula (optionally rotated by a
//! mix angle) or by a 3x3 matrix fitted by the guided calibration.

use log::debug;

/// Raw orientation units per degree convention used by LibrePods:
/// 32000 raw units = 180 degrees.
pub const DEG_PER_RAW: f32 = 180.0 / 32000.0;

/// Orientation in degrees relative to the neutral pose.
#[derive(Debug, Clone, Copy, Default)]
pub struct Orientation {
    /// Pitch (nod up/down) in degrees.
    pub pitch: f32,
    /// Yaw (turn left/right) in degrees.
    pub yaw: f32,
    /// Roll (ear toward shoulder) in degrees. Only produced by a fitted
    /// calibration; the uncalibrated formula cannot separate it (always 0).
    pub roll: f32,
}

/// Acceleration data from the head tracking packet.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct Acceleration {
    pub horizontal: f32,
    pub vertical: f32,
}

/// Linear map from raw channel deltas `(o1, o2, o3)` to `(pitch, yaw, roll)`
/// in degrees. Fitted by the guided calibration; it absorbs the IMU mounting
/// rotation so the head axes no longer leak into each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub m: [[f32; 3]; 3],
}

impl Calibration {
    /// Map raw channel deltas to `[pitch, yaw, roll]` degrees.
    pub fn apply(&self, d: [f32; 3]) -> [f32; 3] {
        let row = |r: [f32; 3]| r[0] * d[0] + r[1] * d[1] + r[2] * d[2];
        [row(self.m[0]), row(self.m[1]), row(self.m[2])]
    }
}

enum Mapping {
    /// LibrePods 2-channel formula, rotated by this angle (radians).
    MixAngle(f32),
    /// Fitted 3-channel calibration.
    Matrix(Calibration),
}

/// The head tracking processor: captures the neutral pose and converts raw
/// packets into orientation.
pub struct HeadTracker {
    mapping: Mapping,
    neutral_sample_count: usize,
    neutral_samples: Vec<[i32; 3]>,
    neutral: Option<[i32; 3]>,
    last_orientation: Option<Orientation>,
    /// Consecutive packets rejected as spikes.
    spike_run: u32,
}

/// Maximum angular change (degrees) between consecutive packets allowed before
/// a reading is considered a spike and discarded.
const MAX_SPIKE_DELTA: f32 = 45.0;

/// A jump that persists for this many packets (~0.2 s) is real motion, not a
/// glitch, and is accepted. Without this bound one large real change would
/// freeze the output on the old value forever.
const MAX_SPIKE_RUN: u32 = 5;

impl HeadTracker {
    /// Uncalibrated tracker using the LibrePods formula rotated by
    /// `mix_angle_deg` (45 = LibrePods).
    pub fn with_mix_angle(neutral_sample_count: usize, mix_angle_deg: f32) -> Self {
        Self::with_mapping(
            neutral_sample_count,
            Mapping::MixAngle(mix_angle_deg.to_radians()),
        )
    }

    /// Tracker driven by a fitted calibration matrix.
    pub fn with_calibration(neutral_sample_count: usize, calibration: Calibration) -> Self {
        Self::with_mapping(neutral_sample_count, Mapping::Matrix(calibration))
    }

    fn with_mapping(neutral_sample_count: usize, mapping: Mapping) -> Self {
        let neutral_sample_count = neutral_sample_count.max(1);
        Self {
            mapping,
            neutral_sample_count,
            neutral_samples: Vec::with_capacity(neutral_sample_count),
            neutral: None,
            last_orientation: None,
            spike_run: 0,
        }
    }

    /// Process a head tracking packet. Returns `None` while the neutral pose
    /// is being captured.
    pub fn process_packet(&mut self, packet: &[u8]) -> Option<(Orientation, Acceleration)> {
        if packet.len() < 55 {
            return None;
        }
        let raw = raw_orientation(packet)?;
        let acceleration = Acceleration {
            horizontal: bytes_to_i16(packet[51], packet[52]) as f32,
            vertical: bytes_to_i16(packet[53], packet[54]) as f32,
        };

        let Some(neutral) = self.neutral else {
            self.neutral_samples.push(raw);
            if self.neutral_samples.len() >= self.neutral_sample_count {
                let n = wrap_mean(&self.neutral_samples);
                debug!("Neutral captured: o1={}, o2={}, o3={}", n[0], n[1], n[2]);
                self.neutral = Some(n);
                self.neutral_samples.clear();
            }
            return None;
        };

        let d = [
            wrap_delta(raw[0], neutral[0]) as f32,
            wrap_delta(raw[1], neutral[1]) as f32,
            wrap_delta(raw[2], neutral[2]) as f32,
        ];
        let current = self.map(d);

        let orientation = match self.last_orientation {
            Some(last)
                if self.spike_run < MAX_SPIKE_RUN
                    && ((current.pitch - last.pitch).abs() > MAX_SPIKE_DELTA
                        || (current.yaw - last.yaw).abs() > MAX_SPIKE_DELTA
                        || (current.roll - last.roll).abs() > MAX_SPIKE_DELTA) =>
            {
                self.spike_run += 1;
                debug!(
                    "Spike filtered ({}/{MAX_SPIKE_RUN}): {current:?}",
                    self.spike_run
                );
                last
            }
            _ => {
                self.spike_run = 0;
                current
            }
        };
        self.last_orientation = Some(orientation);

        Some((orientation, acceleration))
    }

    /// Discard the neutral pose; the next packets capture a new one.
    pub fn reset(&mut self) {
        self.neutral_samples.clear();
        self.neutral = None;
        self.last_orientation = None;
        self.spike_run = 0;
    }

    /// Whether the neutral pose has been captured.
    #[allow(dead_code)]
    pub fn is_calibrated(&self) -> bool {
        self.neutral.is_some()
    }

    fn map(&self, d: [f32; 3]) -> Orientation {
        match &self.mapping {
            Mapping::Matrix(cal) => {
                let [pitch, yaw, roll] = cal.apply(d);
                Orientation { pitch, yaw, roll }
            }
            Mapping::MixAngle(angle) => {
                // At 45 deg this is the LibrePods formula:
                //   pitch = (o2 + o3) / 2   yaw = (o2 - o3) / 2
                let (sin_a, cos_a) = angle.sin_cos();
                let k = std::f32::consts::FRAC_1_SQRT_2 * DEG_PER_RAW;
                Orientation {
                    pitch: (cos_a * d[1] + sin_a * d[2]) * k,
                    yaw: (sin_a * d[1] - cos_a * d[2]) * k,
                    roll: 0.0,
                }
            }
        }
    }
}

/// Raw orientation channels `(o1, o2, o3)` of a head tracking packet.
pub fn raw_orientation(packet: &[u8]) -> Option<[i32; 3]> {
    if packet.len() < 49 {
        return None;
    }
    Some([
        bytes_to_i16(packet[43], packet[44]) as i32,
        bytes_to_i16(packet[45], packet[46]) as i32,
        bytes_to_i16(packet[47], packet[48]) as i32,
    ])
}

/// Difference of two int16 readings, wrapped into [-32768, 32768) so a
/// value crossing the 16-bit seam does not jump by a full turn.
pub fn wrap_delta(a: i32, b: i32) -> i32 {
    (a - b + 32768).rem_euclid(65536) - 32768
}

/// Per-channel mean of int16 readings that is safe across the 16-bit seam.
pub fn wrap_mean(samples: &[[i32; 3]]) -> [i32; 3] {
    let first = samples[0];
    let n = samples.len() as f64;
    std::array::from_fn(|c| {
        let mean_delta = samples
            .iter()
            .map(|s| wrap_delta(s[c], first[c]) as f64)
            .sum::<f64>()
            / n;
        let mean = first[c] + mean_delta.round() as i32;
        wrap_delta(mean, 0)
    })
}

/// Convert two bytes (little-endian) to a signed 16-bit integer.
#[inline]
fn bytes_to_i16(low: u8, high: u8) -> i16 {
    i16::from_le_bytes([low, high])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet_with(raw: [i16; 3]) -> Vec<u8> {
        let mut p = vec![0u8; 70];
        for (i, v) in raw.iter().enumerate() {
            let b = v.to_le_bytes();
            p[43 + 2 * i] = b[0];
            p[44 + 2 * i] = b[1];
        }
        p
    }

    #[test]
    fn neutral_across_seam_does_not_jump() {
        // Neutral straddles the int16 seam; a small move must stay small.
        let mut tracker = HeadTracker::with_mix_angle(2, 45.0);
        assert!(tracker
            .process_packet(&packet_with([0, 32760, 32760]))
            .is_none());
        assert!(tracker
            .process_packet(&packet_with([0, -32760, -32760]))
            .is_none());
        let (o, _) = tracker
            .process_packet(&packet_with([0, -32700, -32700]))
            .unwrap();
        assert!(o.pitch.abs() < 1.0, "pitch jumped: {}", o.pitch);
        assert!(o.yaw.abs() < 1.0, "yaw jumped: {}", o.yaw);
    }

    #[test]
    fn calibrated_matrix_drives_all_three_axes() {
        let cal = Calibration {
            m: [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        };
        let mut tracker = HeadTracker::with_calibration(1, cal);
        assert!(tracker
            .process_packet(&packet_with([100, 200, 300]))
            .is_none());
        let (o, _) = tracker
            .process_packet(&packet_with([110, 220, 330]))
            .unwrap();
        assert_eq!((o.pitch, o.yaw, o.roll), (20.0, 30.0, 10.0));
    }

    /// Identity-like calibration: o2 delta (raw) -> pitch degrees 1:1.
    fn pitch_tracker() -> HeadTracker {
        let cal = Calibration {
            m: [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        let mut tracker = HeadTracker::with_calibration(1, cal);
        assert!(tracker.process_packet(&packet_with([0, 0, 0])).is_none());
        tracker
    }

    #[test]
    fn sustained_jump_is_followed_not_latched() {
        let mut tracker = pitch_tracker();
        tracker.process_packet(&packet_with([0, 10, 0])).unwrap();
        // 10 packets (~0.4 s) of a real, sustained 70 deg move.
        let mut pitch = 0.0;
        for _ in 0..10 {
            pitch = tracker
                .process_packet(&packet_with([0, 80, 0]))
                .unwrap()
                .0
                .pitch;
        }
        assert_eq!(pitch, 80.0, "output stayed latched on the old value");
        // And it keeps tracking afterwards.
        let (o, _) = tracker.process_packet(&packet_with([0, 75, 0])).unwrap();
        assert_eq!(o.pitch, 75.0);
    }

    #[test]
    fn single_packet_glitch_is_still_rejected() {
        let mut tracker = pitch_tracker();
        tracker.process_packet(&packet_with([0, 10, 0])).unwrap();
        let (glitch, _) = tracker.process_packet(&packet_with([0, 90, 0])).unwrap();
        assert_eq!(glitch.pitch, 10.0);
        let (o, _) = tracker.process_packet(&packet_with([0, 12, 0])).unwrap();
        assert_eq!(o.pitch, 12.0);
    }
}
