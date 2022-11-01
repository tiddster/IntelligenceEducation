import igraph

filePath = 'dataset/football.gml'
imagePath = 'dataset/footbalImage.png'


def getValue(value):
    colorList = ['blue','green','purple','yellow','red','pink','orange','black','white','gray','brown','wheat']
    return colorList[int(value)]


g = igraph.Graph.Read_GML(filePath)

g.vs['label'] = ['']

visual_style = dict()
visual_style['vertex_color'] = list(map(getValue, (g.vs['value'])))
visual_style['bbox'] = [0, 0, 800, 800]

image = igraph.plot(g, **visual_style)
image.save(imagePath)

