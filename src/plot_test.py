from matplotlib.pyplot import plot, draw, show
import matplotlib.pyplot as plt


def make_plot():
    plot([1,2,3])
    draw()
    print 'Plot displayed, waiting for it to be closed.'

print('Do something before plotting.')
# Now display plot in a window
make_plot()
# This line was moved up <----
show()

answer = raw_input('Back to main after plot window closed? ')
if answer == 'y':
    print('Move on')
else:
    print('Nope')
plt.ioff()