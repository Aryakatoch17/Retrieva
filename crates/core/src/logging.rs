//! Tracing subscriber initialization.
//!
//! OpenTelemetry OTLP export is available behind the `otel` feature.

use crate::config::LoggingConfig;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Options for [`init_logging`].
#[derive(Debug, Clone)]
pub struct LoggingOptions {
    /// Filter directive, e.g. `info` or `retrieva=debug,tower=warn`.
    pub level: String,
    /// `pretty` or `json`.
    pub format: String,
}

impl From<&LoggingConfig> for LoggingOptions {
    fn from(value: &LoggingConfig) -> Self {
        Self {
            level: value.level.clone(),
            format: value.format.clone(),
        }
    }
}

impl Default for LoggingOptions {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "pretty".into(),
        }
    }
}

/// Install a global tracing subscriber.
///
/// When the `otel` feature is enabled and `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
/// spans are also exported via OTLP (tonic). Safe to call once at process start.
pub fn init_logging(opts: &LoggingOptions) -> Result<(), crate::RetrievaError> {
    let filter = EnvFilter::try_new(&opts.level).unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(feature = "otel")]
    {
        return init_with_otel(opts, filter);
    }

    #[cfg(not(feature = "otel"))]
    {
        let result = match opts.format.as_str() {
            "json" => tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().json())
                .try_init(),
            _ => tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().pretty())
                .try_init(),
        };
        result.map_err(|e| {
            crate::RetrievaError::Internal(format!("logging already initialized: {e}"))
        })?;
        Ok(())
    }
}

#[cfg(feature = "otel")]
fn init_with_otel(opts: &LoggingOptions, filter: EnvFilter) -> Result<(), crate::RetrievaError> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::Layer;

    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
    let fmt_layer = match opts.format.as_str() {
        "json" => fmt::layer().json().boxed(),
        _ => fmt::layer().pretty().boxed(),
    };

    if let Some(endpoint) = endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| crate::RetrievaError::Internal(format!("otlp exporter: {e}")))?;
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();
        let tracer = provider.tracer("retrieva");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(otel_layer)
            .try_init()
            .map_err(|e| {
                crate::RetrievaError::Internal(format!("logging already initialized: {e}"))
            })?;
        tracing::info!("OpenTelemetry OTLP export enabled");
        // Leak provider so it lives for process lifetime (simple Phase 8 approach).
        std::mem::forget(provider);
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| {
                crate::RetrievaError::Internal(format!("logging already initialized: {e}"))
            })?;
        tracing::debug!("otel feature on but OTEL_EXPORTER_OTLP_ENDPOINT unset; fmt only");
    }
    Ok(())
}
