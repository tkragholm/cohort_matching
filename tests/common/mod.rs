//! Synthetic cohort and risk-set policy shaped like a register-study caller.
//!
//! Shared by the golden tests and the benchmark, through the public API only.

#![allow(dead_code)]

use chrono::{Months, NaiveDate, TimeDelta};
use cohort_matching::constraints::caliper_on_field;
use cohort_matching::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub struct Lcg(u64);

impl Lcg {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Fast hasher for membership-only id sets.
#[derive(Default)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(word))
                .wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
}

pub type IdSet<'a> = HashSet<&'a str, BuildHasherDefault<IdHasher>>;
pub type IdMap<'a, V> = HashMap<&'a str, V, BuildHasherDefault<IdHasher>>;

pub struct Person {
    pub id: String,
    pub birth: NaiveDate,
    pub sex: &'static str,
    pub municipality: String,
    pub region: String,
    pub family: String,
    pub first_date: Option<NaiveDate>,
    pub death: Option<NaiveDate>,
    pub emigration: Option<NaiveDate>,
    pub exposed: bool,
    pub eligible_until: Option<NaiveDate>,
    pub parent_year: Option<f64>,
}

/// A cohort of `n` people of whom about `cases_per_mille` per thousand are exposed.
pub fn cohort(n: usize, cases_per_mille: u64, municipalities: u64, seed: u64) -> Vec<Person> {
    let mut rng = Lcg::new(seed);
    let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    (0..n)
        .map(|i| {
            // Ids that do not sort in cohort order.
            let id = format!(
                "{:010}",
                (i as u64).wrapping_mul(2_654_435_761) % 9_999_999_967
            );
            let birth = base + TimeDelta::days(rng.below(5_800) as i64);
            let municipality = rng.below(municipalities);
            let exposed_draw = rng.below(1000);
            let exposed = exposed_draw < cases_per_mille;
            let other_date = rng.below(1000) < 50;
            let age_years = 1 + rng.below(5) as u32;
            let first_date = if exposed || other_date {
                Some(
                    birth
                        .checked_add_months(Months::new(age_years * 12))
                        .unwrap()
                        + TimeDelta::days(rng.below(300) as i64),
                )
            } else {
                None
            };
            let death = (rng.below(1000) < 20)
                .then(|| birth + TimeDelta::days(400 + rng.below(4_000) as i64));
            let emigration = (rng.below(1000) < 30)
                .then(|| birth + TimeDelta::days(400 + rng.below(4_000) as i64));
            let eligible_until = (rng.below(1000) < 30)
                .then(|| birth + TimeDelta::days(400 + rng.below(4_000) as i64));
            let parent_year = (rng.below(1000) >= 20).then(|| 1960.0 + rng.below(40) as f64);
            Person {
                id,
                birth,
                sex: if rng.below(2) == 0 { "1" } else { "2" },
                municipality: format!("{:03}", 100 + municipality),
                region: format!("R{}", municipality % 5),
                family: format!("F{}", rng.below((n as u64 / 2).max(1))),
                first_date,
                death,
                emigration,
                exposed,
                eligible_until,
                parent_year,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
pub enum Level {
    Municipality,
    Region,
}

pub type Rec = RoleTransitionRecord<BaseRecord>;

pub fn records(people: &[Person], level: Level, with_family: bool, with_caliper: bool) -> Vec<Rec> {
    people
        .iter()
        .map(|p| {
            let mut r = BaseRecord::new(&p.id, p.birth);
            r.strata.insert("sex".to_string(), p.sex.to_string());
            match level {
                Level::Municipality => {
                    r.strata
                        .insert("municipality".to_string(), p.municipality.clone());
                }
                Level::Region => {
                    r.strata.insert("region".to_string(), p.region.clone());
                }
            }
            if with_family {
                r.strata.insert("family".to_string(), p.family.clone());
            }
            if with_caliper && let Some(y) = p.parent_year {
                r = r.with_numeric("parent_year", y);
            }
            if let Some(d) = p.death {
                r = r.with_death_date(d);
            }
            if let Some(d) = p.emigration {
                r = r.with_emigration_date(d);
            }
            RoleTransitionRecord::from_record(r, p.first_date)
        })
        .collect()
}

pub struct PassPolicy<'a> {
    pub unexposed_ids: &'a IdSet<'a>,
    pub eligible_until: &'a IdMap<'a, NaiveDate>,
    pub matched_case_ids: &'a IdSet<'a>,
    pub used_control_ids: &'a IdSet<'a>,
}

impl RiskSetPolicy<Rec> for PassPolicy<'_> {
    fn is_case_eligible(&self, row: &Rec, age_limit_years: u8) -> bool {
        !self.unexposed_ids.contains(row.id())
            && !self.matched_case_ids.contains(row.id())
            && DefaultRiskSetPolicy.is_case_eligible(row, age_limit_years)
    }

    fn is_control_at_risk(&self, case: &Rec, control: &Rec, case_event_date: NaiveDate) -> bool {
        !self.used_control_ids.contains(control.id())
            && self
                .eligible_until
                .get(control.id())
                .is_none_or(|until| *until > case_event_date)
            && DefaultRiskSetPolicy.is_control_at_risk(case, control, case_event_date)
    }
}

pub struct Sets<'a> {
    pub unexposed: IdSet<'a>,
    pub eligible_until: IdMap<'a, NaiveDate>,
}

pub fn sets(people: &[Person]) -> Sets<'_> {
    Sets {
        unexposed: people
            .iter()
            .filter(|p| !p.exposed)
            .map(|p| p.id.as_str())
            .collect(),
        eligible_until: people
            .iter()
            .filter_map(|p| p.eligible_until.map(|d| (p.id.as_str(), d)))
            .collect(),
    }
}

/// Two passes shaped like a register-study caller: municipality, then region
/// for the cases the first pass left unmatched.
pub fn two_pass(people: &[Person], seed: u64, with_caliper: bool) -> (MatchOutcome, MatchOutcome) {
    let s = sets(people);
    let none = IdSet::default();
    let ratio = MatchRatio::new(4).unwrap();
    let fallback = vec![MatchRatio::new(2).unwrap(), MatchRatio::new(1).unwrap()];
    let window = BirthDateWindowDays::new(30).unwrap();
    let age = AgeLimitYears::new(6).unwrap();
    let calipers = || {
        if with_caliper {
            vec![caliper_on_field::<Rec>("parent_year", 5.0)]
        } else {
            Vec::new()
        }
    };
    let resident = |r: &Rec, d: NaiveDate| r.is_resident_at(d);

    let muni_cohort = records(people, Level::Municipality, false, with_caliper);
    let pass1 = MatchJob::new_transition(&muni_cohort, seed)
        .with_ratio(ratio)
        .with_ratio_fallback(fallback.clone())
        .with_birth_window(window)
        .with_age_limit(age)
        .with_exact_matches(["sex", "municipality"])
        .with_alive_check()
        .with_resident_check_by(resident)
        .with_constraints(calipers())
        .with_risk_set_policy(PassPolicy {
            unexposed_ids: &s.unexposed,
            eligible_until: &s.eligible_until,
            matched_case_ids: &none,
            used_control_ids: &none,
        })
        .run();
    let matched: IdSet<'_> = pass1.pairs.iter().map(|p| p.case_id.as_str()).collect();
    let used: IdSet<'_> = pass1.pairs.iter().map(|p| p.control_id.as_str()).collect();

    let region_cohort = records(people, Level::Region, false, with_caliper);
    let pass2 = MatchJob::new_transition(&region_cohort, seed.wrapping_add(1))
        .with_ratio(ratio)
        .with_ratio_fallback(fallback)
        .with_birth_window(window)
        .with_age_limit(age)
        .with_exact_matches(["sex", "region"])
        .with_alive_check()
        .with_resident_check_by(resident)
        .with_constraints(calipers())
        .with_risk_set_policy(PassPolicy {
            unexposed_ids: &s.unexposed,
            eligible_until: &s.eligible_until,
            matched_case_ids: &matched,
            used_control_ids: &used,
        })
        .run();
    (pass1, pass2)
}
