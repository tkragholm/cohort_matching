use chrono::NaiveDate;
use cohort_matching::{
    BaseRecord, Constraint, DeterministicSelection, MatchingCriteria, StandardMatchRequest,
    match_standard,
};

fn main() {
    let anchors = vec![BaseRecord::new(
        "a",
        NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date"),
    )];
    let candidates = anchors.clone();
    let criteria = MatchingCriteria::default();

    let constraints: Vec<Box<dyn Constraint<BaseRecord>>> = Vec::new();
    let request = StandardMatchRequest::new(&criteria, DeterministicSelection, &constraints);
    let _ = match_standard(&anchors, &candidates, request);
}
