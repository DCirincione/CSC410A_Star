# CSC410 A* Assignment

This repository contains my implementation of the A* (A-star) search algorithm in Python for CSC410.  
The program reads a graph from a '.data' file, runs the A* algorithm with a Euclidean heuristic,  
and prints the explored order, the shortest path, and the total path cost.
# Input File Rules/Format
- The **first** city listed is the **start** node.  
- The **last** city listed is the **goal** node.  
- Blank lines and lines beginning with `#` are ignored.  
- Coordinates may be integers or floats.  
- Labels must be alphabetic characters (e.g., `A`, `B`, `Goal`). 

## Command to run the program
python3 a_star.py inputs/example-1.data
python3 a_star.py inputs/example2-1.data
python3 a_star.py inputs/example-1.data --show-debug
python3 a_star.py inputs/example2-1.data --show-debug

