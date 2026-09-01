from IPython.display import Image, display
from builder import create_graph

graph = create_graph()

display(Image(graph.get_graph().draw_mermaid_png()))