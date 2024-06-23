COM=g++
VER=-std=c++2a

output: main.o net.o
	$(COM) $(VER) main.o net.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp

net.o: ./src/net.cpp
	$(COM) $(VER) -c ./src/net.cpp