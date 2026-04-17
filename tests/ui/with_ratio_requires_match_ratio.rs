use chrono::NaiveDate;
use cohort_matching::MatchJob;

fn main() {
    let anchors = vec![cohort_matching::BaseRecord::new(
        "a",
        NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date"),
    )];
    let candidates = anchors.clone();

    let _ = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_ratio(1)
        .run();
}
