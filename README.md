<h1 align="center">
The Ultimate Class Warfare Resolver

## Description
Fire Emblem is a series of ~14 RPG games with a LOT of characters, and a lot of different classes
This project exists to basically find the optimal class for any unit according to their stats and growths
(Honestly it's not really important, I just wanted to play around with this data to understand the math behind RPG unit design)

Data Source: [serenesforest](https://serenesforest.net)


## Trivia
- Not all games use the same stats, so yeah, fun
- We need to remove the character's base class stats to get their absolute personal ones, fun too
- I'll be forced to vectorize everything because of power creep
- I'll probably have to do a PCA on classes, because "Flier" is basically Pegasus Knight with a mustache (and will also reduce the number of categories to predict)
- Should probably add additional boolean features to class according to gameplay purpose too
- FE13 will be our testing (not validation) set since its characters don't have base classes
- FE5 is the hardest game in the series, and it was also by far the hardest to clean. All my homies hate FE5.
- I'll probably do a config file at some point. But right now I'm just hungry, so PR time it is
