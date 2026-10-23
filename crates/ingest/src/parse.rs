//! File parsers for supported ingest formats.

use retrieva_core::RetrievaError;
use std::path::Path;

/// Extract plain text from a supported file path.
///
/// Supported extensions: `.txt`, `.md`, `.markdown`, `.pdf`.
pub fn extract_text(path: &Path) -> Result<String, RetrievaError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "txt" | "md" | "markdown" => read_utf8(path),
        "pdf" => extract_pdf(path),
        other => Err(RetrievaError::Ingest(format!(
            "unsupported file type '.{other}' for {}",
            path.display()
        ))),
    }
}

/// Whether `path` has a known ingestible extension.
pub fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "txt" | "md" | "markdown" | "pdf"
    )
}

fn read_utf8(path: &Path) -> Result<String, RetrievaError> {
    std::fs::read_to_string(path)
        .map_err(|e| RetrievaError::Ingest(format!("failed to read {}: {e}", path.display())))
}

fn extract_pdf(path: &Path) -> Result<String, RetrievaError> {
    pdf_extract::extract_text(path)
        .map_err(|e| RetrievaError::Ingest(format!("failed to parse PDF {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn reads_plain_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("note.txt");
        let mut f = fs::File::create(&path).expect("create");
        writeln!(f, "hello retrieva").expect("write");
        let text = extract_text(&path).expect("extract");
        assert!(text.contains("hello retrieva"));
    }

    #[test]
    fn rejects_unknown_extension() {
        let err = extract_text(Path::new("file.bin")).expect_err("bin");
        assert!(matches!(err, RetrievaError::Ingest(_)));
    }
}
