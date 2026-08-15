use std::time::Duration;

/// Per-step breakdown of `EthPirServer::init_with_timed`.
#[derive(Clone, Copy, Debug, Default)]
pub struct InitTimings {
    pub keyword_index: Duration,
    pub records_scatter: Duration,
    pub server_alloc: Duration,
    pub database_encode: Duration,
    pub query_mask: Duration,
    pub offline: Duration,
    pub staging_alloc: Duration,
}

impl InitTimings {
    pub fn total(&self) -> Duration {
        self.keyword_index
            + self.records_scatter
            + self.server_alloc
            + self.database_encode
            + self.query_mask
            + self.offline
            + self.staging_alloc
    }
}

/// Per-step breakdown of a database rebuild.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefreshTimings {
    pub database_encode: Duration,
    pub precompute: Duration,
    pub install: Duration,
}

impl RefreshTimings {
    pub fn total(&self) -> Duration {
        self.database_encode + self.precompute + self.install
    }
}

/// Per-step breakdown of `EthPirServer::rebuild_keyword_index`.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeywordRebuildTimings {
    pub collect_keys: Duration,
    pub mphf_rebuild: Duration,
    pub permute: Duration,
    pub refresh: RefreshTimings,
}

impl KeywordRebuildTimings {
    pub fn total(&self) -> Duration {
        self.collect_keys + self.mphf_rebuild + self.permute + self.refresh.total()
    }
}

/// Where a running server's memory goes, in bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryReport {
    pub serving_database: usize,
    pub staging_database: usize,
    pub precomputation: usize,
    pub online_scratch_pool: usize,
    pub records: usize,
    pub keyword_directory: usize,
}

impl MemoryReport {
    pub fn total(&self) -> usize {
        self.serving_database
            + self.staging_database
            + self.precomputation
            + self.online_scratch_pool
            + self.records
            + self.keyword_directory
    }

    pub fn refresh_peak(&self) -> usize {
        self.total() + self.precomputation
    }
}
