//! Stable chunk → ANN key mapping.

use sha2::{Digest, Sha256};

/// Map a chunk id string to a stable `u64` ANN key (first 8 bytes of SHA-256).
pub fn chunk_key(chunk_id: &str) -> u64 {
    let digest = Sha256::digest(chunk_id.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_across_calls() {
        assert_eq!(chunk_key("doc#0"), chunk_key("doc#0"));
        assert_ne!(chunk_key("doc#0"), chunk_key("doc#1"));
    }
}
