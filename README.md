# SchoolOfAI
Assignments for school of AI


#methodOfFiniteDifferences.py:
hello!

This is my reinforcement algorithm for a bipedal walker, for the midterm assignment of the Move 37 Course.
I have created an algorithm based on the method of Finite Differences, 
which was explained in this video https://www.youtube.com/watch?v=2P2Dj5PX5cg&t=415s by Colin Skow (lesson 5.2).
One of the things I did different, is that I only update the weights
if the reward with these weights is higher than the reward with the current weights.

For now, there are 2000 iterations. This can be adjusted at row 42.

There are a few things I did implement, but are not turned on in the .py in this repository.
The first is a learning rate which goes down as time goes by. This means that learning in the beginning relies more on the present,
whereas learning later on relies more on the past. You can turn this on by setting decay to something other than 0 (row 40 in the .py)
The second is what I call a goCrazy-functionality, 
and this entails that if the weights were not updated for x amount of iterations, they will update anyway. 
I have made this as I noticed that the algorithm started out updating a lot, and hoped this would get it over that bump. 
However, the results were a bit disappointing, as after this update, the weights would get up a bit, but not higher than before. 
But feel free to try it yourself! Turn this on by setting the goCrazy to a lower number (row44 in the .py).

Have fun!
