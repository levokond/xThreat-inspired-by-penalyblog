I will add improvements as I go:

 - Initially, the move probability was defined as just the average pass accuracy in the dataset, ignoring whether the pass was made in their own half, where it would usually be easier to complete a pass, or in the opposition half, where it would be more difficult to do so in tighter spaces. So I decided to make the move probability dynamic instead of a fixed number, i.e., by calculating pass accuracy for the "grid" (the pass is added to a location based on the starting point).
