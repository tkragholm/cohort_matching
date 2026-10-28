//! Pins the outputs of representative matchings: pairs in order, counts,
//! diagnostics and weights, each reduced to a 64-bit FNV-1a hash of the debug
//! rendering. A change to the engine that moves any pair, weight, drop count or
//! the order of the output changes a hash.

mod common;

use chrono::NaiveDate;
use cohort_matching::prelude::*;
use cohort_matching::{DeterministicSelection, NearestBirthDateSelection};
use common::{Level, Person, Rec, cohort, records, two_pass};
use std::collections::HashMap;

fn fnv(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn outcome_hash(outcome: &MatchOutcome) -> u64 {
    fnv(&format!("{outcome:?}"))
}

fn weights_hash(outcome: &MatchOutcome) -> u64 {
    let unit = match_weights_from_pairs(
        &outcome.pairs,
        MatchWeightMethod::AnchorUnitCandidateFractional,
    )
    .to_table();
    let pair = pair_weights_from_pairs(
        &outcome.pairs,
        MatchWeightMethod::AnchorUnitCandidateFractional,
    )
    .to_table();
    fnv(&format!("{unit:?}{pair:?}"))
}

fn check(name: &str, outcome: &MatchOutcome, expected: (u64, u64), nonempty: bool) {
    let got = (outcome_hash(outcome), weights_hash(outcome));
    if nonempty {
        assert!(
            !outcome.pairs.is_empty(),
            "{name}: no pairs, test is vacuous"
        );
    }
    println!(
        "GOLDEN {name} pairs={} matched={} unmatched={} => ({:#018x}, {:#018x})",
        outcome.pairs.len(),
        outcome.matched_cases,
        outcome.unmatched_cases,
        got.0,
        got.1
    );
    if std::env::var_os("GOLDEN_RECORD").is_none() {
        assert_eq!(got, expected, "{name}");
    }
}

#[test]
fn two_pass_municipality_then_region() {
    let people = cohort(6_000, 90, 12, 11);
    let (p1, p2) = two_pass(&people, 7, false);
    assert!(p2.matched_cases > 0, "region pass must do work");
    check("tp_pass1", &p1, E_TP_PASS1, true);
    check("tp_pass2", &p2, E_TP_PASS2, true);
}

#[test]
fn two_pass_with_numeric_caliper() {
    let people = cohort(6_000, 90, 12, 12);
    let (p1, p2) = two_pass(&people, 21, true);
    check("tpc_pass1", &p1, E_TPC_PASS1, true);
    check("tpc_pass2", &p2, E_TPC_PASS2, true);
}

#[test]
fn sparse_strata_many_municipalities() {
    // Few people per stratum: many cases find no comparator.
    let people = cohort(3_000, 120, 90, 13);
    let (p1, p2) = two_pass(&people, 3, false);
    assert!(p1.unmatched_cases > 0);
    check("sparse_pass1", &p1, E_SPARSE_PASS1, true);
    check("sparse_pass2", &p2, E_SPARSE_PASS2, true);
}

fn transition(
    people: &[Person],
    level: Level,
    keys: &[&str],
    with_family: bool,
    unique: Option<&str>,
    nearest: bool,
) -> MatchOutcome {
    let cohort_records = records(people, level, with_family, false);
    let job = MatchJob::new_transition(&cohort_records, 99)
        .with_ratio(MatchRatio::new(3).unwrap())
        .with_birth_window(BirthDateWindowDays::new(60).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(keys.iter().copied())
        .with_unique_by_key(unique.map(str::to_string))
        .with_alive_check();
    if nearest {
        job.with_strategy(NearestBirthDateSelection).run()
    } else {
        job.run()
    }
}

#[test]
fn default_policy_nearest_selection() {
    let people = cohort(4_000, 100, 10, 14);
    let out = transition(
        &people,
        Level::Municipality,
        &["sex", "municipality"],
        false,
        None,
        true,
    );
    check("nearest", &out, E_NEAREST, true);
}

#[test]
fn unique_by_strata_key() {
    let people = cohort(4_000, 100, 10, 15);
    let out = transition(
        &people,
        Level::Municipality,
        &["sex", "municipality"],
        true,
        Some("family"),
        false,
    );
    check("unique_family", &out, E_UNIQUE_FAMILY, true);
}

#[test]
fn no_strata_at_all() {
    let people = cohort(2_000, 100, 10, 16);
    let out = transition(&people, Level::Region, &[], false, None, false);
    check("no_strata", &out, E_NO_STRATA, true);
}

#[test]
fn required_key_missing_on_some_records() {
    // Half the records carry no "municipality" at all, and "absent" must not
    // equal "absent" differently than before.
    let people = cohort(4_000, 100, 10, 17);
    let mut recs: Vec<Rec> = records(&people, Level::Municipality, false, false);
    for (i, r) in recs.iter_mut().enumerate() {
        if i % 3 == 0 {
            r.record.strata.remove("municipality");
        }
        if i % 7 == 0 {
            r.record.strata.remove("sex");
        }
    }
    let out = MatchJob::new_transition(&recs, 5)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(45).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .run();
    check("missing_key", &out, E_MISSING_KEY, true);
}

fn standard_inputs(seed: u64) -> (Vec<BaseRecord>, Vec<BaseRecord>) {
    let people = cohort(5_000, 200, 8, seed);
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for p in &people {
        let mut r = BaseRecord::new(&p.id, p.birth);
        r.strata.insert("sex".to_string(), p.sex.to_string());
        r.strata
            .insert("municipality".to_string(), p.municipality.clone());
        r.strata.insert("family".to_string(), p.family.clone());
        if p.exposed {
            anchors.push(r);
        } else {
            candidates.push(r);
        }
    }
    // Candidate ids that also appear as anchor ids exercise the self-match rule.
    for a in anchors.iter().take(20) {
        let mut twin = a.clone();
        twin.birth_date += chrono::TimeDelta::days(1);
        candidates.push(twin);
    }
    (anchors, candidates)
}

#[test]
fn standard_without_replacement_with_fallback() {
    let (anchors, candidates) = standard_inputs(18);
    let out = MatchJob::new_standard(&anchors, &candidates, 4)
        .with_ratio(MatchRatio::new(5).unwrap())
        .with_ratio_fallback(vec![
            MatchRatio::new(3).unwrap(),
            MatchRatio::new(1).unwrap(),
        ])
        .with_birth_window(BirthDateWindowDays::new(20).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .run();
    check("std_norepl", &out, E_STD_NOREPL, true);
}

#[test]
fn standard_with_replacement() {
    let (anchors, candidates) = standard_inputs(19);
    let out = MatchJob::new_standard(&anchors, &candidates, 8)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(30).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .with_replacement(true)
        .run();
    check("std_repl", &out, E_STD_REPL, true);
}

#[test]
fn standard_unique_by_key() {
    let (anchors, candidates) = standard_inputs(20);
    let out = MatchJob::new_standard(&anchors, &candidates, 6)
        .with_ratio(MatchRatio::new(3).unwrap())
        .with_birth_window(BirthDateWindowDays::new(40).unwrap())
        .with_exact_matches(["sex"])
        .with_unique_by_key(Some("family".to_string()))
        .run();
    check("std_unique", &out, E_STD_UNIQUE, true);
}

#[test]
fn standard_no_strata_deterministic() {
    let (anchors, candidates) = standard_inputs(21);
    let out = MatchJob::new_standard(&anchors, &candidates, 6)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(10).unwrap())
        .with_strategy(DeterministicSelection)
        .run();
    check("std_nostrata", &out, E_STD_NOSTRATA, true);
}

#[test]
fn hash_is_sensitive_to_order() {
    // The golden hashes see pair order: swapping two pairs changes them.
    let people = cohort(1_000, 100, 4, 22);
    let (mut p1, _) = two_pass(&people, 1, false);
    let before = outcome_hash(&p1);
    p1.pairs.swap(0, 1);
    assert_ne!(before, outcome_hash(&p1));
    let _: HashMap<u8, NaiveDate> = HashMap::new();
}

const E_TP_PASS1: (u64, u64) = (0x63f4b43829ef2f07, 0xba2d20b850211f2b);
const E_TP_PASS2: (u64, u64) = (0x12e30a49de49df54, 0x9c388a3ffb90ed91);
const E_TPC_PASS1: (u64, u64) = (0x7c87c38a1205f328, 0xabba74f8048e91ac);
const E_TPC_PASS2: (u64, u64) = (0xb9f5e4200994a7fe, 0x1e44c4a02f2fe9c8);
const E_SPARSE_PASS1: (u64, u64) = (0x3d0802c359c3ad3b, 0x8d6598c02353c98a);
const E_SPARSE_PASS2: (u64, u64) = (0x30151d03d11a1dc4, 0xb02aaddd03c1b311);
const E_NEAREST: (u64, u64) = (0x959f7432847ef008, 0x585c1833c1e022cf);
const E_UNIQUE_FAMILY: (u64, u64) = (0xba373c07d1999ccf, 0x5bf2edc754923f1f);
const E_NO_STRATA: (u64, u64) = (0x9f2d9e72ec6062ed, 0xb3af339dd86f2543);
const E_MISSING_KEY: (u64, u64) = (0x4ba905545e7cefcb, 0x49b20c1163143340);
const E_STD_NOREPL: (u64, u64) = (0x8f2cdd879d49d7d4, 0x98f0339e11d9ad0e);
const E_STD_REPL: (u64, u64) = (0x9fd9853ccfe82116, 0x9e3273668b942a2e);
const E_STD_UNIQUE: (u64, u64) = (0xeea7887eb73ad616, 0xa262ee78f0e2c5db);
const E_STD_NOSTRATA: (u64, u64) = (0xfbe1e86e4ad4936d, 0x1a8b1221f7c395a3);
