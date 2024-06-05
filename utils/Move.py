from utils.Matrix import Matrix

class Move:
  def up(game):
    game = Matrix.transpose(game)
    game, done = Matrix.cover_up(game)
    game, done, score = Matrix.merge(game, done)
    game = Matrix.cover_up(game)[0]
    game = Matrix.transpose(game)
    return game, done, score

  def down(game):
    game = Matrix.reverse(Matrix.transpose(game))
    game, done = Matrix.cover_up(game)
    game, done, score = Matrix.merge(game, done)
    game = Matrix.cover_up(game)[0]
    game = Matrix.transpose(Matrix.reverse(game))
    return game, done, score

  def left(game):
    game, done = Matrix.cover_up(game)
    game, done, score = Matrix.merge(game, done)
    game = Matrix.cover_up(game)[0]
    return game, done, score

  def right(game):
    game = Matrix.reverse(game)
    game, done = Matrix.cover_up(game)
    game, done, score = Matrix.merge(game, done)
    game = Matrix.cover_up(game)[0]
    game = Matrix.reverse(game)
    return game, done, score