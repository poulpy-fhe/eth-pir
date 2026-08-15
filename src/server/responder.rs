use super::{EthPirServer, Serving};
use crate::{EthPirError, EthQuery, EthResponse, RecordCodec};

/// Cloneable database-service handle.
pub struct EthPirResponder {
    pub(super) serving: Serving,
}

impl Clone for EthPirResponder {
    fn clone(&self) -> Self {
        Self {
            serving: self.serving.clone(),
        }
    }
}

impl EthPirResponder {
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        self.try_respond(query)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError> {
        respond_with(&self.serving, query)
    }

    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        self.try_respond_batch(queries)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError> {
        respond_batch_with(&self.serving, queries)
    }

    pub fn try_respond_bytes(&self, query: &[u8]) -> Result<Vec<u8>, EthPirError> {
        let mut serving = self
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)?;
        let query = EthQuery::read_from(&mut { query }, serving.params(), serving.layout())?;
        let response = serving.try_respond(&query)?;
        let mut bytes = Vec::new();
        response.write_to(serving.params().module(), &mut bytes)?;
        Ok(bytes)
    }

    /// Answer many serialized queries in one pass over the database.
    ///
    /// One result per input, in order. A query that fails to parse yields its own
    /// error rather than failing the batch — one malformed request from one
    /// client must not deny everyone else in the same window.
    ///
    /// The outer error is reserved for a failure of the batch itself, which no
    /// individual query can be blamed for.
    pub fn try_respond_batch_bytes<B: AsRef<[u8]>>(
        &self,
        queries: &[B],
    ) -> Result<Vec<Result<Vec<u8>, EthPirError>>, EthPirError> {
        let mut serving = self
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)?;

        // Parse first, keeping each query's position so results line up with
        // inputs even though the malformed ones never reach the database.
        let mut parsed = Vec::with_capacity(queries.len());
        let mut slots = Vec::with_capacity(queries.len());
        let mut out: Vec<Result<Vec<u8>, EthPirError>> = Vec::with_capacity(queries.len());
        for query in queries {
            match EthQuery::read_from(&mut { query.as_ref() }, serving.params(), serving.layout()) {
                Ok(q) => {
                    slots.push(out.len());
                    parsed.push(q);
                    out.push(Ok(Vec::new()));
                }
                Err(e) => out.push(Err(e.into())),
            }
        }

        if parsed.is_empty() {
            return Ok(out);
        }

        let responses = serving.try_respond_batch(&parsed)?;
        for (slot, response) in slots.into_iter().zip(responses) {
            let mut bytes = Vec::new();
            out[slot] = response
                .write_to(serving.params().module(), &mut bytes)
                .map(|()| bytes)
                .map_err(EthPirError::from);
        }
        Ok(out)
    }
}

impl<C: RecordCodec> EthPirServer<C> {
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        self.try_respond(query)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError> {
        respond_with(&self.serving, query)
    }

    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        self.try_respond_batch(queries)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError> {
        respond_batch_with(&self.serving, queries)
    }

    pub fn responder(&self) -> EthPirResponder {
        EthPirResponder {
            serving: self.serving.clone(),
        }
    }
}

fn respond_with(serving: &Serving, query: &EthQuery) -> Result<EthResponse, EthPirError> {
    serving
        .lock()
        .map_err(|_| EthPirError::ServerPoisoned)?
        .try_respond(query)
        .map_err(EthPirError::from)
}

fn respond_batch_with(
    serving: &Serving,
    queries: &[EthQuery],
) -> Result<Vec<EthResponse>, EthPirError> {
    serving
        .lock()
        .map_err(|_| EthPirError::ServerPoisoned)?
        .try_respond_batch(queries)
        .map_err(EthPirError::from)
}
