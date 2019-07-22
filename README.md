# _manta_ ![manta](https://github.com/ramellose/manta/blob/master/manta.png)

Microbial association network clustering algorithm.

[![Build Status](https://travis-ci.com/ramellose/manta.svg?token=9mhqeTh13MErxyrk5zR8&branch=master)](https://travis-ci.com/ramellose/manta)

This toolbox is aimed at weighted & undirected microbial networks. It uses a diffusion-based proccess to carry out network clustering.
Moreover, it can generate a Cytoscape-compatible layout that places taxonomically similar & structurally similar nodes closer together.
Contact the author at lisa.rottjers (at) kuleuven.be. Your feedback is much appreciated!
This version is still in early beta and has been tested for Python 3.6.

## Getting Started

To install _manta_, run:
```
pip install git+https://github.com/ramellose/manta.git
```

To run the script, only two arguments are required: input and output filepaths.
The script recognizes gml, graphml and cyjs files by their extension. By default, cyjs is exported.
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

For a demo run, a network generated from fecal samples of bats has been included.

This data was downloaded from [QIITA](https://qiita.ucsd.edu/study/description/11815).

Lutz, H. L., Jackson, E. W., Dick, C., Webala, P. W., Babyesiza, W. S., Peterhans, J. C. K., ... & Gilbert, J. A. (2018). __Associations between Afrotropical bats, parasites, and microbial symbionts.__ bioRxiv, 340109.

To run the demo, run _manta_ as follows:
```
manta -i demo -o filepath_to_output_network
```

For an elaborate demo that goes through more variables, go [here](https://ramellose.github.io/networktutorials/demo.html).

For documentation of specific functions, check out [the Sphinx documentation](https://ramellose.github.io/manta/index.html).
### Contributions

This software is still in early alpha. Any feedback or bug reports will be much appreciated!

## Authors

* **Lisa Röttjers** - [ramellose](https://github.com/ramellose)
* **Karoline Faust** - [hallucigenia-sparsa](https://github.com/hallucigenia-sparsa)

See also the list of [contributors](https://github.com/ramellose/manca/contributors) who participated in this project.

## License

This project is licensed under the Apache License - see the [LICENSE.txt](LICENSE.txt) file for details


