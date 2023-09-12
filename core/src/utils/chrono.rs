use crate::common::duration::*;

/// Parse a string into a chrono DateTime
///
/// If the time portion is not supplied in the input string, the time will be
/// default to 12:00:00am UTC time
pub fn parse_date_time_string(d: &str) -> Option<chrono::DateTime<chrono::Utc>> {
  let midnight_naive = chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap();
  dateparser::parse_with(d, &chrono::Utc, midnight_naive).ok()
}

/// Parse a string into a chrono Duration
pub fn parse_duration_string(d: &str) -> Option<Duration> {
  parse_relative_duration::parse(d).ok().map(|d| Duration(d))
}
