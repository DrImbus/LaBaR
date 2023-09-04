# LaBaR (Laue Back Reflection)
Given a crystal (i.e. the crystal parameters as well as the basis) this programm calculates the position and intensity of Laue Back 
Reflection Spots on a flat screen some distance from the crystal.

It is supposed to be used in combination with the Cologne Laue Indexation Program (CLIP) written by Olaf Schumann.
It gives the user the ability to check the orientation of a crystal given by CLIP.

The result is outputted as a csv file as well as an image.
The orientation of the crystal is determined by 3 Euler-Angles (order of rotation is Z-, X-, Y-Rotation) but more rotations may be appended 
to properly orientate the crystal.
