/// Parse a string into a chrono DateTime
pub fn parse_date_time_string(d: &str) -> Option<chrono::DateTime<chrono::Utc>> {
  dateparser::parse(d).ok()
}

/// Parse a string into a chrono Duration
pub fn parse_duration_string(d: &str) -> Option<chrono::Duration> {
  let d1 = parse_duration::parse(d).ok()?;
  chrono::Duration::from_std(d1).ok()
}
