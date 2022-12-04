CXX := g++
CXXFLAGS := -std=c++17 -Wall -O3
LDFLAGS := -lpthread

.PHONY: all run clean
all: hw2.cpp
	$(CXX) $(CXXFLAGS) -o hw2 hw2.cpp $(LDFLAGS)

run: all
	./hw2

clean:
	-rm -f hw2