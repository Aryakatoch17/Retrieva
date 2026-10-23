//! Word-based text chunking with configurable size and overlap.

use unicode_segmentation::UnicodeSegmentation;

/// Split `text` into overlapping chunks sized by Unicode word count.
///
/// Empty / whitespace-only input yields an empty list. When `text` fits in a
/// single window, one chunk is returned.
pub fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    debug_assert!(chunk_size > 0);
    debug_assert!(chunk_overlap < chunk_size);

    let words: Vec<&str> = text.unicode_words().collect();
    if words.is_empty() {
        return Vec::new();
    }

    if words.len() <= chunk_size {
        return vec![join_words(&words)];
    }

    let step = chunk_size - chunk_overlap;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        chunks.push(join_words(&words[start..end]));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

fn join_words(words: &[&str]) -> String {
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("", 8, 2).is_empty());
        assert!(chunk_text("   \n\t  ", 8, 2).is_empty());
    }

    #[test]
    fn short_text_is_single_chunk() {
        let chunks = chunk_text("one two three", 10, 2);
        assert_eq!(chunks, vec!["one two three"]);
    }

    #[test]
    fn overlapping_windows() {
        // 10 words, size 4, overlap 2 → step 2
        let text = "a b c d e f g h i j";
        let chunks = chunk_text(text, 4, 2);
        assert_eq!(chunks, vec!["a b c d", "c d e f", "e f g h", "g h i j",]);
    }

    #[test]
    fn last_partial_window_kept() {
        let text = "a b c d e";
        let chunks = chunk_text(text, 3, 1);
        assert_eq!(chunks, vec!["a b c", "c d e"]);
    }
}
