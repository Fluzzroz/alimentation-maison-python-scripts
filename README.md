# alimentation-maison-python-scripts

In this branch, I'm trying to add the option of specifying the first meeting location as an input.
However, when I add it, the algorithm either does not converge (even when I specify the first meeting as per the solution found when that option was not available), or the times are not respected. I also noticed that only "bordering nodes" are violated. For instance, consider this schedule:
- Node @ Time
- 1   @ 11:00
- 2   @ 11:00
- 3   @ 11:00
- 4   @ 12:00
- 5   @ 12:00
- 6   @ 12:00
- 7   @ 13:00
- 8   @ 13:00
- 9   @ 13:00
 
 In this example, only the 4 and the 7 will be violated, and the violation will construct routes as if 4 was at 11:00 and 7 at 12:00.
 I have no clue as to why this is happening. At this point I suspect a bug in the or-tools, but I could be wrong.
