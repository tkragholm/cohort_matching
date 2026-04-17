use chrono::NaiveDate;
use cohort_matching::{
    BaseRecord, Constraint, DeterministicSelection, MatchingCriteria, RoleTransitionRecord,
    TransitionMatchRequest, match_transition,
};

fn main() {
    let records = vec![
        RoleTransitionRecord::from_record(
            BaseRecord::new(
                "case",
                NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date"),
            ),
            Some(NaiveDate::from_ymd_opt(2014, 1, 1).expect("valid date")),
        ),
        RoleTransitionRecord::from_record(
            BaseRecord::new(
                "candidate",
                NaiveDate::from_ymd_opt(2010, 1, 2).expect("valid date"),
            ),
            None,
        ),
    ];

    let criteria = MatchingCriteria::default();
    let options = cohort_matching::RoleTransitionOptions::default();
    let constraints: Vec<Box<dyn Constraint<RoleTransitionRecord<BaseRecord>>>> = Vec::new();

    let request =
        TransitionMatchRequest::new(&criteria, &options, DeterministicSelection, &constraints);
    let _ = match_transition(&records, request);
}
