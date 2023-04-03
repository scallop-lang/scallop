# The Scallop book

Scallop book aims to provide a comprehensive guide on Scallop and its toolchain.
This book is mostly written in markdown and will be compiled and built by `mdbook`.

## For contribution

To develop the book, please first install the `mdbook` (assuming you have Rust and `cargo` properly installed):

``` bash
$ cargo install mdbook
```

Then, type the following command to start serving the book at localhost:

``` bash
$ make serve-book
```

While the server is running, go to your browser and type `localhost:3000` to start reading the book.
Note that when you edit the markdown files in the source, the page on your browser should automatically refresh.
