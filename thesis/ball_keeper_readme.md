### TL;DR ###


```python
import ball_keeper
...
g = NEAT.Genome(...)

game = ball_keeper.init()
features = ball_keeper.evaluateGenome(game, g)
```

### How to use extracted ball_keeper game ###

First, import the game into your script
```python
import ball_keeper
```

Then call
```python
ball_keeper.init()
```
and it returns a `game` tuple you can use for these functions:

* `playHuman(game)` if you want to manually play.
* `playGenome(game, genome)` to watch a genome play.
* `evaluateGenome(game, genome)` to evaluate a genome without visuals. (obviously waay faster)

Please reuse the game tuple between calls to spare yourself the overhead of re-initialising the game


In the original example, the outside loop looked like this:

```python
for generation in range(1000):
        ...

        for i, g in enumerate(genome_list):
            total_fitness = 0
            for trial in range(20):
                f, fast_mode = evaluate(g, space, screen, fast_mode, rnd.randint(80, 400), rnd.randint(-200, 200), rnd.randint(80, 400))
                total_fitness += f
            g.SetFitness(total_fitness / 20)
        ...
```

You can replace the call to` evaluate(..)` with the new `ball_keeper.evaluateGenome(..)` function, and it should work similar.

The play/evaluate... functions now return a tuple of extracted features, not a single fitness value. (If you want, you can use one of the values alone as a classic fitness value.)

The current features extracted are, in order:
* \# of timesteps passed
* \# of times ball and player collided
* \# of player jumps
* Total distance traveled by player
* Total distance traveled by ball
* Highest ball velocity reached
