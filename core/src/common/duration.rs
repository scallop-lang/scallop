use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(pub chronoutil::RelativeDuration);

impl Duration {
  pub fn years(y: i32) -> Self {
    Self(chronoutil::RelativeDuration::years(y))
  }

  pub fn months(y: i32) -> Self {
    Self(chronoutil::RelativeDuration::months(y))
  }

  pub fn days(d: i64) -> Self {
    Self(chronoutil::RelativeDuration::days(d))
  }
}

impl std::fmt::Display for Duration {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.0.months == 0 {
      self.0.duration.fmt(f)
    } else {
      f.write_fmt(format_args!("R{}MO {}", self.0.months, self.0.duration))
    }
  }
}

impl From<chronoutil::RelativeDuration> for Duration {
  fn from(value: chronoutil::RelativeDuration) -> Self {
    Self(value)
  }
}

impl From<chrono::Duration> for Duration {
  fn from(value: chrono::Duration) -> Self {
    let rela_dura = chronoutil::RelativeDuration::from(value);
    Self::from(rela_dura)
  }
}

impl std::ops::Add<Duration> for Duration {
  type Output = Duration;

  fn add(self, rhs: Duration) -> Self::Output {
    Self(self.0 + rhs.0)
  }
}

impl std::ops::Sub<Duration> for Duration {
  type Output = Duration;

  fn sub(self, rhs: Duration) -> Self::Output {
    Self(self.0 - rhs.0)
  }
}

impl std::ops::Add<Duration> for chrono::DateTime<chrono::Utc> {
  type Output = chrono::DateTime<chrono::Utc>;

  fn add(self, rhs: Duration) -> Self::Output {
    self + rhs.0
  }
}

impl std::ops::Sub<Duration> for chrono::DateTime<chrono::Utc> {
  type Output = chrono::DateTime<chrono::Utc>;

  fn sub(self, rhs: Duration) -> Self::Output {
    self - rhs.0
  }
}

impl std::ops::Mul<i32> for Duration {
  type Output = Duration;

  fn mul(self, rhs: i32) -> Self::Output {
    Self(self.0 * rhs)
  }
}

impl std::ops::Div<i32> for Duration {
  type Output = Duration;

  fn div(self, rhs: i32) -> Self::Output {
    Self(self.0 / rhs)
  }
}

impl Serialize for Duration {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::*;
    let mut state = serializer.serialize_struct("Duration", 1)?;
    state.serialize_field("months", &self.0.months)?;
    state.serialize_field("duration", &self.0.duration.num_seconds())?;
    state.end()
  }
}
