# manca

Microbial association network clustering algorithm. Version 0.1.0.

This toolbox is aimed at weighted & undirected microbial networks. It uses a diffusion-based proccess to carry out network clustering & centrality calculations.
Moreover, it can generate a Cytoscape-compatible layout that places taxonomically similar & structurally similar nodes closer together.
Contact the author at lisa.rottjers (at) kuleuven.be. Your feedback is much appreciated!
This version is still in early alpha and has been tested for Python 3.6.

## Getting Started

To install <i>manca</i>, run:
```
pip install git+https://github.com/ramellose/manca.git
```

To run the script, only two arguments are required:
```
manca -i filepath_to_input_network -o filepath_to_output_network
```

To generate a taxonomically-informed layout and centrality scores, add some flags:
```
manca -i filepath_to_input_network -o filepath_to_output_network -f cyjs --layout --central -tax filepath_to_tax_table
```

Layouts can only be generated for .cyjs network files, which are selected with the flag -f cyjs.

For a complete explanation of all the parameters, run:
```
manca -h
```

### Contributions

This software is still in early alpha. Any feedback or bug reports will be much appreciated!

## Authors

* **Lisa Röttjers** - [ramellose](https://github.com/ramellose)

See also the list of [contributors](https://github.com/ramellose/manca/contributors) who participated in this project.

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details


