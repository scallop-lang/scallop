# Scallop GPT

## Installation

To install from source, please use the following command:

``` bash
$ make -C etc/scallopy-plugins install-gpt
```

For developers, you can do

``` bash
$ make -C etc/scallopy-plugins develop-gpt
```

As such, you can edit in-source and have the changes reflected in your execution.

## Usage

The Scallop GPT plugin provides the following constructs:

- (Foreign Attribute) `@gpt`
- (Foreign Attribute) `@gpt_extract_info`
- (Foreign Attribute) `@gpt_encoder`
- (Foreign Predicate) `gpt(bound input: String, output: String)`
- (Foreign Function) `$gpt(String) -> String`

### Foreign Attribute `@gpt`

### Foreign Attribute `@gpt_extract_info`

### Foreign Attribute `@gpt_encoder`

### Foreign Predicate `gpt`

### Foreign Function `$gpt`
