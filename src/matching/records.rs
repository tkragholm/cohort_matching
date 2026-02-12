use crate::types::{BaseRecord, ParticipantRecord, RoleSwitchingRecord, RoleTransitionRecord};
use chrono::NaiveDate;
use std::collections::HashMap;

pub trait MatchingRecord {
    fn id(&self) -> &str;
    fn birth_date(&self) -> NaiveDate;
    fn strata(&self) -> &HashMap<String, String>;
    fn unique_key(&self) -> Option<&str>;
}

pub trait RoleIndexedRecord: MatchingRecord {
    fn event_date(&self) -> Option<NaiveDate>;
}

impl MatchingRecord for BaseRecord {
    fn id(&self) -> &str {
        &self.id
    }

    fn birth_date(&self) -> NaiveDate {
        self.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.strata
    }

    fn unique_key(&self) -> Option<&str> {
        self.unique_key.as_deref()
    }
}

impl MatchingRecord for ParticipantRecord {
    fn id(&self) -> &str {
        self.core.id.as_str()
    }

    fn birth_date(&self) -> NaiveDate {
        self.core.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.core.strata
    }

    fn unique_key(&self) -> Option<&str> {
        self.core.unique_key.as_deref()
    }
}

impl MatchingRecord for RoleSwitchingRecord {
    fn id(&self) -> &str {
        self.participant.core.id.as_str()
    }

    fn birth_date(&self) -> NaiveDate {
        self.participant.core.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.participant.core.strata
    }

    fn unique_key(&self) -> Option<&str> {
        self.participant.core.unique_key.as_deref()
    }
}

impl RoleIndexedRecord for RoleSwitchingRecord {
    fn event_date(&self) -> Option<NaiveDate> {
        self.diagnosis_date
    }
}

impl<R: MatchingRecord> MatchingRecord for RoleTransitionRecord<R> {
    fn id(&self) -> &str {
        self.record.id()
    }

    fn birth_date(&self) -> NaiveDate {
        self.record.birth_date()
    }

    fn strata(&self) -> &HashMap<String, String> {
        self.record.strata()
    }

    fn unique_key(&self) -> Option<&str> {
        self.record.unique_key()
    }
}

impl<R: MatchingRecord> RoleIndexedRecord for RoleTransitionRecord<R> {
    fn event_date(&self) -> Option<NaiveDate> {
        self.transition_date
    }
}
