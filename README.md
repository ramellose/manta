# manta

Microbial association network clustering algorithm. Version 0.1.0.

[![Build Status](https://travis-ci.com/ramellose/manta.svg?token=9mhqeTh13MErxyrk5zR8&branch=master)](https://travis-ci.com/ramellose/manta)

This toolbox is aimed at weighted & undirected microbial networks. It uses a diffusion-based proccess to carry out network clustering & centrality calculations.
Moreover, it can generate a Cytoscape-compatible layout that places taxonomically similar & structurally similar nodes closer together.
Contact the author at lisa.rottjers (at) kuleuven.be. Your feedback is much appreciated!
This version is still in early alpha and has been tested for Python 3.6.

## Getting Started

To install <i>manta</i>, run:
```
pip install git+https://github.com/ramellose/manta.git
```

To run the script, only two arguments are required: input and output filepaths.
The script recognizes gml, graphml and cyjs files. By default, cyjs is exported.
It also accepts text files with edge lists, with the third column containing edge weight.
```
manta -i filepath_to_input_network -o filepath_to_output_network
```

To generate a taxonomically-informed layout, add some flags:
```
manta -i filepath_to_input_network -o filepath_to_output_network -f cyjs --layout --central -tax filepath_to_tax_table
```

Layouts can only be generated for .cyjs network files, which are selected with the flag -f cyjs.

For a complete explanation of all the parameters, run:
```
manta -h
```

Alternatively, check out the Sphinx documentation @ https://ramellose.github.io/manta/index.html.
### Contributions

This software is still in early alpha. Any feedback or bug reports will be much appreciated!

## Authors

* **Lisa Röttjers** - [ramellose](https://github.com/ramellose)

See also the list of [contributors](https://github.com/ramellose/manca/contributors) who participated in this project.

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.txt) file for details


