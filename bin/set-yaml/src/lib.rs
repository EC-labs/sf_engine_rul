use std::{fs, error::Error};
use yaml_rust::{yaml, yaml::Yaml};


pub struct Config {
    key: String,
    value: String,
    filename: String,
}

impl Config {

    pub fn new<T>(mut args: T) -> Result<Config, &'static str>
    where 
        T: Iterator<Item=String>
    {
        args.next();
        let filename = match args.next() {
            Some(s) => s, 
            None => return Err("Missing filename positional argument")
        };
        let key = match args.next() {
            Some(s) => s, 
            None => return Err("Missing key positional argument")
        };
        let value = match args.next() {
            Some(s) => s, 
            None => return Err("Missing key positional argument")
        };
        Ok(Config { filename, key, value })
    }
}

#[derive(Debug)]
enum Selector {
    Index(usize),
    Key(String),
}

enum ParseState {
    Start,
    Ready,
    ReadingIndex, 
    ReadingKey,
}

fn walk_selectors(mut yml: yaml::Yaml, selectors: Vec<Selector>) -> Result<yaml::Yaml, String> {
    let mut value = &mut yml; 
    for selector in selectors.iter() {
        value = match selector {
            Selector::Key(k) => {
                match value {
                    Yaml::Hash(h) => {
                        let key = Yaml::String(k.to_string());
                        match h.get_mut(&key) {
                            Some(v) => v,
                            None => {
                                return Err(format!(
                                    "Key does not exist in current yaml structure:",
                                ));
                            }
                        }
                    },
                    _ => { 
                        return Err(format!(
                            "Cannot apply key on current yaml structure:\n{:?}", 
                            value
                        ));
                    }
                }
            }, 
            Selector::Index(i) => { 
                match value {
                    Yaml::Array(a) => match a.get_mut(*i) {
                        Some(v) => v,
                        None => 
                            return Err(format!(
                                "Index does not exist in Array:\n", 
                            ))
                    },
                    _ => {
                        return Err(format!(
                            "Index cannot be applied to current yaml structure:\n{:?}",
                            value
                        ));
                    }
                }
            }
        };
    }
    Ok(yml)
}

fn parse_yaml_path(path: &str) -> Result<Vec<Selector>, &'static str> {
    let char_iter = path.chars();
    let mut parse_state = ParseState::Start;
    let mut acc = String::new();
    let mut selectors: Vec<Selector> = Vec::new();
    for c in char_iter {
        parse_state = match parse_state {
            ParseState::Start => match c {
                '.' | 'a'..='z' | 'A'..='Z' | '_' => {
                    if c != '.' {
                        acc = acc + &c.to_string();
                    }
                    ParseState::ReadingKey
                },
                '[' => ParseState::ReadingIndex,
                _ => return Err("Incorrect format")
            }
            ParseState::Ready => match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    acc = acc + &c.to_string();
                    ParseState::ReadingKey
                }
                '[' => ParseState::ReadingIndex,
                _ => return Err("Incorrect format"),
            },
            ParseState::ReadingKey => match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    acc = acc + &c.to_string();
                    ParseState::ReadingKey
                },
                '.' => {
                    selectors.push(Selector::Key(acc.parse::<String>().unwrap()));
                    acc.clear();
                    ParseState::Ready
                }
                '[' => {
                    selectors.push(Selector::Key(acc.parse::<String>().unwrap()));
                    acc.clear();
                    ParseState::ReadingIndex
                }
                _ => return Err("Incorrect format"),
            }, 
            ParseState::ReadingIndex => match c {
                '0'..='9' => {
                    acc = acc + &c.to_string();
                    ParseState::ReadingIndex
                }
                ']' => {
                    selectors.push(Selector::Index(acc.parse::<usize>().unwrap()));
                    acc.clear();
                    ParseState::Ready
                }
                _ => return Err("Incorrect format"),
            },
        }
    }
    match parse_state {
        ParseState::Start => return Err("empty key"),
        ParseState::ReadingIndex => {
            selectors.push(Selector::Index(acc.parse::<usize>().unwrap()));
        }
        ParseState::ReadingKey => {
            selectors.push(Selector::Key(acc.parse::<String>().unwrap()));
        },
        ParseState::Ready => {}
    }
    Ok(selectors)
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>>{
    let contents = fs::read_to_string(&config.filename).unwrap();
    let mut contents = yaml::YamlLoader::load_from_str(&contents)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    // let contents = & mut contents;
    //println!("{:?}", contents);
    // match contents {
    //     Yaml::Hash(ref mut h) => {
    //         let k = Yaml::String(String::from("batch_size"));
    //         let v = Yaml::Integer(20);
    //         h.insert(k, v);
    //     },
    //     _ => println!("---2---")
    // }
    let selectors = parse_yaml_path(&config.key)?;
    walk_selectors(contents, selectors)?;
    Ok(())
}
