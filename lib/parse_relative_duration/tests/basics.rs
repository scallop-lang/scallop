extern crate num;
extern crate chrono;
extern crate chronoutil;
extern crate parse_relative_duration;

use num::BigInt;
use chrono::Duration;
use chronoutil::RelativeDuration;

use parse_relative_duration::parse;

macro_rules! test_parse {
    (fn $fun:ident($string: expr, $months: expr, $seconds: expr, $nanoseconds: expr)) => {
        #[test]
        fn $fun() {
            assert_eq!(parse($string), Ok(
                RelativeDuration::months($months).with_duration(Duration::seconds($seconds) + Duration::nanoseconds($nanoseconds))
            ))
        }
    };
}

macro_rules! test_invalid {
    (fn $fun:ident($string: expr, $error: expr)) => {
        #[test]
        fn $fun() {
            assert_eq!(parse($string), Err($error));
        }
    };
}

test_parse!(fn nano1("1nsec", 0, 0, 1));
test_parse!(fn nano2("1ns", 0, 0, 1));
test_parse!(fn nano_dec("1.07 ns", 0, 0, 1));
test_parse!(fn nano_exp1("1.07e5 ns", 0, 0, 107_000));
test_parse!(fn nano_exp2("1.07e+5 ns", 0, 0, 107_000));
test_parse!(fn nano_exp3("1.07e-5 ns", 0, 0, 0));
test_parse!(fn nano_exp4("1e5 ns", 0, 0, 100_000));
test_parse!(fn nano_exp5("1e+5 ns", 0, 0, 100_000));
test_parse!(fn nano_exp6("1e-5 ns", 0, 0, 0));
test_parse!(fn nano_neg("-1nsec", 0, 0, -1));

test_parse!(fn micro1("1usec", 0, 0, 1_000));
test_parse!(fn micro2("1us", 0, 0, 1_000));
test_parse!(fn micro_dec("1.07 us", 0, 0, 1_070));
test_parse!(fn micro_exp1("1.07e5 us", 0, 0, 107_000_000));
test_parse!(fn micro_exp2("1.07e+5 us", 0, 0, 107_000_000));
test_parse!(fn micro_exp3("1.07e-5 us", 0, 0, 0));
test_parse!(fn micro_exp4("1e5 us", 0, 0, 100_000_000));
test_parse!(fn micro_exp5("1e+5 us", 0, 0, 100_000_000));
test_parse!(fn micro_exp6("1e-5 us", 0, 0, 0));
test_parse!(fn micro_neg("-1usec", 0, 0, -1_000));

test_parse!(fn milli1("1msec", 0, 0, 1_000_000));
test_parse!(fn milli2("1ms", 0, 0, 1_000_000));
test_parse!(fn milli_dec("1.07 ms", 0, 0, 1_070_000));
test_parse!(fn milli_exp1("1.07e5 ms", 0, 107, 0));
test_parse!(fn milli_exp2("1.07e+5 ms", 0, 107, 0));
test_parse!(fn milli_exp3("1.07e-5 ms", 0, 0, 10));
test_parse!(fn milli_exp4("1e5 ms", 0, 100, 0));
test_parse!(fn milli_exp5("1e+5 ms", 0, 100, 0));
test_parse!(fn milli_exp6("1e-5 ms", 0, 0, 10));
test_parse!(fn mili_neg("-1msec", 0, 0, -1_000_000));

test_parse!(fn sec1("1seconds", 0, 1, 0));
test_parse!(fn sec2("1second", 0, 1, 0));
test_parse!(fn sec3("1sec", 0, 1, 0));
test_parse!(fn sec4("1s", 0, 1, 0));
test_parse!(fn sec_dec("1.07 s", 0, 1, 70_000_000));
test_parse!(fn sec_exp1("1.07e5 s", 0, 107_000, 0));
test_parse!(fn sec_exp2("1.07e+5 s", 0, 107_000, 0));
test_parse!(fn sec_exp3("1.07e-5 s", 0, 0, 10_700));
test_parse!(fn sec_exp4("1e5 s", 0, 100_000, 0));
test_parse!(fn sec_exp5("1e+5 s", 0, 100_000, 0));
test_parse!(fn sec_exp6("1e-5 s", 0, 0, 10_000));
test_parse!(fn sec_neg("-1seconds", 0, -1, 0));

test_parse!(fn min1("1minutes", 0, 60, 0));
test_parse!(fn min2("1minute", 0, 60, 0));
test_parse!(fn min3("1min", 0, 60, 0));
test_parse!(fn min4("1m", 0, 60, 0));
test_parse!(fn min_dec("1.07 m", 0, 64, 200_000_000));
test_parse!(fn min_exp1("1.07e5 m", 0, 6_420_000, 0));
test_parse!(fn min_exp2("1.07e+5 m", 0, 6_420_000, 0));
test_parse!(fn min_exp3("1.07e-5 m", 0, 0, 642_000));
test_parse!(fn min_exp4("1e5 m", 0, 6_000_000, 0));
test_parse!(fn min_exp5("1e+5 m", 0, 6_000_000, 0));
test_parse!(fn min_exp6("1e-5 m", 0, 0, 600_000));
test_parse!(fn min_neg("-1minutes", 0, -60, 0));

test_parse!(fn hour1("1hours", 0, 3_600, 0));
test_parse!(fn hour2("1hour", 0, 3_600, 0));
test_parse!(fn hour3("1hr", 0, 3_600, 0));
test_parse!(fn hour4("1h", 0, 3_600, 0));
test_parse!(fn hour_dec("1.07 h", 0, 3_852, 0));
test_parse!(fn hour_exp1("1.07e5 h", 0, 385_200_000, 0));
test_parse!(fn hour_exp2("1.07e+5 h", 0, 385_200_000, 0));
test_parse!(fn hour_exp3("1.07e-5 h", 0, 0, 38_520_000));
test_parse!(fn hour_exp4("1e5 h", 0, 360_000_000, 0));
test_parse!(fn hour_exp5("1e+5 h", 0, 360_000_000, 0));
test_parse!(fn hour_exp6("1e-5 h", 0, 0, 36_000_000));
test_parse!(fn hour_neg("-1hours", 0, -3_600, 0));

test_parse!(fn day1("1days", 0, 86_400, 0));
test_parse!(fn day2("1day", 0, 86_400, 0));
test_parse!(fn day3("1d", 0, 86_400, 0));
test_parse!(fn day_dec("1.07 d", 0, 92_448, 0));
test_parse!(fn day_exp1("1.07e5 d", 0, 9_244_800_000, 0));
test_parse!(fn day_exp2("1.07e+5 d", 0, 9_244_800_000, 0));
test_parse!(fn day_exp3("1.07e-5 d", 0, 0, 924_480_000));
test_parse!(fn day_exp4("1e5 d", 0, 8_640_000_000, 0));
test_parse!(fn day_exp5("1e+5 d", 0, 8_640_000_000, 0));
test_parse!(fn day_exp6("1e-5 d", 0, 0, 864_000_000));
test_parse!(fn day_neg("-1days", 0, -86_400, 0));

test_parse!(fn week1("1weeks", 0, 604_800, 0));
test_parse!(fn week2("1week", 0, 604_800, 0));
test_parse!(fn week3("1w", 0, 604_800, 0));
test_parse!(fn week_dec("1.07 w", 0, 647_136, 0));
test_parse!(fn week_exp1("1.07e5 w", 0, 64_713_600_000, 0));
test_parse!(fn week_exp2("1.07e+5 w", 0, 64_713_600_000, 0));
test_parse!(fn week_exp3("1.07e-5 w", 0, 6, 471_360_000));
test_parse!(fn week_exp4("1e5 w", 0, 60_480_000_000, 0));
test_parse!(fn week_exp5("1e+5 w", 0, 60_480_000_000, 0));
test_parse!(fn week_exp6("1e-5 w", 0, 6, 48_000_000));
test_parse!(fn week_neg("-1weeks", 0, -604_800, 0));

test_parse!(fn month1("1months", 1, 0, 0));
test_parse!(fn month2("1month", 1, 0, 0));
test_parse!(fn month3("1M", 1, 0, 0));
test_parse!(fn month_dec("1.07 M", 0, 2_813_828, 220_000_000));
test_parse!(fn month_exp1("1.07e5 M", 0, 281_382_822_000, 0));
test_parse!(fn month_exp2("1.07e+5 M", 0, 281_382_822_000, 0));
test_parse!(fn month_exp3("1.07e-5 M", 0, 28, 138_282_200));
test_parse!(fn month_exp4("1e5 M", 0, 262_974_600_000, 0));
test_parse!(fn month_exp5("1e+5 M", 0, 262_974_600_000, 0));
test_parse!(fn month_exp6("1e-5 M", 0, 26, 297_460_000));
test_parse!(fn month_neg("-1 months", -1, 0, 0));

test_parse!(fn year1("1years", 12, 0, 0));
test_parse!(fn year2("1year", 12, 0, 0));
test_parse!(fn year3("1y", 12, 0, 0));
test_parse!(fn year_dec("1.07 y", 0, 33_765_938, 640_000_000));
test_parse!(fn year_exp1("1.07e5 y", 0, 3_376_593_864_000, 0));
test_parse!(fn year_exp2("1.07e+5 y", 0, 3_376_593_864_000, 0));
test_parse!(fn year_exp3("1.07e-5 y", 0, 337, 659_386_400));
test_parse!(fn year_exp4("1e5 y", 0, 3_155_695_200_000, 0));
test_parse!(fn year_exp5("1e+5 y", 0, 3_155_695_200_000, 0));
test_parse!(fn year_exp6("1e-5 y", 0, 315, 569_520_000));
test_parse!(fn year_neg("-1 years", -12, 0, 0));

test_parse!(fn multi_with_space("1min    10 seconds", 0, 70, 0));
test_parse!(fn multi_no_space("1min10seconds", 0, 70, 0));
test_parse!(fn multi_out_of_order("10year1min10seconds5h", 120, 18_070, 0));
test_parse!(fn multi_repetition("1min 10 minute", 0, 660, 0));

test_parse!(fn multiple_units("16 min seconds", 0, 960, 0));

test_parse!(fn negatives("1 day -15 minutes", 0, 85_500, 0));
test_parse!(fn unmatched_negatives("1 day - 15 minutes", 0, 87_300, 0));
test_parse!(fn negative_duration("-3 days 71 hours", 0, -3600, 0));

test_parse!(fn no_unit("15", 0, 15, 0));
test_parse!(fn no_unit_with_noise(".:++++]][][[][15[]][][]:}}}}", 0, 15, 0));

test_invalid!(fn invalid_int("1e11232345982734592837498234 years", parse::Error::ParseInt("11232345982734592837498234".to_string())));
test_invalid!(fn invalid_unit("16 sdfwe", parse::Error::UnknownUnit("sdfwe".to_string())));
test_invalid!(fn no_value("year", parse::Error::NoValueFound("year".to_string())));
test_invalid!(fn wrong_order("year15", parse::Error::NoUnitFound("15".to_string())));

#[test]
fn number_too_big() {
    assert_eq!(
        Ok(parse("123456789012345678901234567890 seconds")),
        "123456789012345678901234567890"
            .parse::<BigInt>()
            .map(|int| Err(parse::Error::OutOfBounds(int)))
    );
}

test_invalid!(fn not_enough_units("16 17 seconds", parse::Error::NoUnitFound("16".to_string())));
