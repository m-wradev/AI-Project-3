default: id3.cpp
	g++ id3.cpp -o id3

dbg_bldtree: id3.cpp
	g++ id3.cpp -o id3 -D DBG_BLDTREE

dbg_gain: id3.cpp
	g++ id3.cpp -o id3 -D DBG_GAIN

dbg_ident: id3.cpp
	g++ id3.cpp -o id3 -D DBG_IDENT

dbg_all: id3.cpp
	g++ id3.cpp -o id3 -D DBG_GAIN -D DBG_BLDTREE -D DBG_IDENT
