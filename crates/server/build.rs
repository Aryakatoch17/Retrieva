fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vendored protoc so CI / Windows hosts need no system protobuf-compiler.
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);

    let proto = "../../proto/retrieva/v1/retrieva.proto";
    let include = "../../proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto], &[include])?;
    Ok(())
}
