use yaml_rust::yaml;

use set_yaml::Config; 


fn main() {
    let config = Config::new(std::env::args()).unwrap_or_else(|e| {
        eprintln!("{:?}", e);
        std::process::exit(1);
    });
    set_yaml::run(config).unwrap_or_else(|e| {
        eprintln!("{:?}", e);
        std::process::exit(1);
    });

}
