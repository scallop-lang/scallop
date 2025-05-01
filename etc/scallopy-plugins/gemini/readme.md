# Scallop Gemini

This implementation is entirely based on the Scallop GPT Plugin code.
The main goal of this adaptation is to make the use for small tasks more accessible.

## Installation

To install from source, please use the following command:

``` bash
$ make -C etc/scallopy-plugins install-gemini
```

For developers, you can do

``` bash
$ make -C etc/scallopy-plugins develop-gemini
```

As such, you can edit in-source and have the changes reflected in your execution.

## Usage

The Scallop Gemini plugin provides the following constructs:

- (Foreign Attribute) `@gemini_extract_info`
- (Foreign Attribute) `@gemini_encoder`
- (Foreign Predicate) `gemini(bound input: String, output: String)`
- (Foreign Function) `$gemini(String) -> String`

### Foreign Attribute `@gemini_extract_info`

### Foreign Attribute `@gemini_encoder`

### Foreign Predicate `gemini`

### Foreign Function `$gemini`
