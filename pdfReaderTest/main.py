# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# importing required modules
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import fitz
import re
import ave

class Page:

    def __init__(self, embeds, sntcs, dstIndxs):
        self.embedings = embeds
        self.sentences = sntcs
        self.distance_indx = dstIndxs


class Doc:

    def __init__(self):
        self.dist_indxs = None
        self.distance_indx = None
        self.page_averages = None
        self.doc = None
        self.path = None
        self._threshold_intersection = 0.9  # if the intersection is large enough.
        self.page_data = []
        self.AVE = None

    def loadpdf(self, path):

        self.path = path

        self.doc = fitz.open(path)

    def get_embed_data(self, pages = None,  shortest_string=10):


        for i, page in enumerate(self.doc):

            page_txt = page.get_text()

            #text = re.sub("https?:\/\/.*[\r\n]*", "", page_txt)
            #text = ''.join(i for i in text if not i.isdigit())
            #text = text.replace("Yvonne Marshall and Benjamin Alberti", "")
            #text = text.replace("A Matter of Difference: Karen Barad, Ontology and Archaeological Bodies", "")
            #text = text.replace("Downloaded from", "")

            sentences = page_txt.split(". ")

            #sentences = list(filter(lambda s: len(s) > shortest_string, sentences))

            embeds = self.embedings(sentences)

            distance_indx = self.find_distances(embeds)

            pageData = Page(embeds, sentences, distance_indx)
            #

            if self.page_data is None:

                self.page_data = [pageData]
            else:
                self.page_data.append(pageData)

    def highlight_divergent(self, number_of_divergent=4):

        for page, data in zip(self.doc, self.page_data):

            for dist, indx in data.distance_indx[-number_of_divergent:]:

                ### SEARCH
                text = "Sample text"
                text_instances = page.search_for(data.sentences[indx])

                ### HIGHLIGHT
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()

    def save(self, name="output"):
        ### OUTPUT
        self.doc.save(name + ".pdf", garbage=4, deflate=True, clean=True)

    def read_highlights(self, colour=None):

        # if no colour set
        if colour is None:
            # itterate over pages with enumerator
            for i, page in enumerate(self.doc):
                # get the annotations of that page
                annot = page.annots()
                print(len(list(annot)), "len")
                # list to store the co-ordinates of all highlights
                highlights = []

                # while loop
                while True:

                    try:
                        # get next annotation
                        this_annot = annot.__next__()

                        # if it is a highlighting annot
                        if this_annot.type[0] == 8:
                            # get coordinates
                            all_coordinates = this_annot.vertices

                            # coordinates have 4 verts
                            if len(all_coordinates) == 4:
                                # get coordinates as a rect
                                highlight_coord = fitz.Quad(all_coordinates).rect
                                # append them to the list
                                highlights.append(highlight_coord)
                            else:
                                # break the coordinates into sets of 4
                                all_coordinates = [all_coordinates[x:x + 4] for x in range(0, len(all_coordinates), 4)]
                                # loop over new coordinates
                                for i in range(0, len(all_coordinates)):
                                    # get rect coords and add to list
                                    coord = fitz.Quad(all_coordinates[i]).rect
                                    highlights.append(coord)

                            #debbugging trying to work out
                            highlight = page.add_highlight_annot(highlights[0])
                            highlight.set_colors(colors='Red')
                            highlight.update()

                            all_words = page.get_text_words()
                            # List to store all the highlighted texts
                            highlight_text = []
                            for h in highlights[0]:
                                sentence = [w[4] for w in all_words if fitz.Rect(w[0:4]).intersect(h)]
                                highlight_text.append(" ".join(sentence))

                    except StopIteration:
                        break

    def _check_contain(self, r_word, points):
        """If `r_word` is contained in the rectangular area.

        The area of the intersection should be large enough compared to the
        area of the given word.

        Args:
            r_word (fitz.Rect): rectangular area of a single word.
            points (list): list of points in the rectangular area of the
                given part of a highlight.

        Returns:
            bool: whether `r_word` is contained in the rectangular area.
        """
        # `r` is mutable, so everytime a new `r` should be initiated.
        r = fitz.Quad(points).rect
        r.intersect(r_word)

        if r.getArea() >= r_word.getArea() * self._threshold_intersection:
            contain = True
        else:
            contain = False
        return contain

    def _extract_annot(self, annot, words_on_page):
        """Extract words in a given highlight.

        Args:
            annot (fitz.Annot): [description]
            words_on_page (list): [description]

        Returns:
            str: words in the entire highlight.
        """
        quad_points = annot.vertices
        quad_count = int(len(quad_points) / 4)
        sentences = ['' for i in range(quad_count)]
        for i in range(quad_count):
            points = quad_points[i * 4: i * 4 + 4]
            words = [
                w for w in words_on_page if
                self._check_contain(fitz.Rect(w[:4]), points)
            ]
            sentences[i] = ' '.join(w[4] for w in words)
        sentence = ' '.join(sentences)
        print(sentence)
        return sentence

    def find_distances(self, embedings):

        average = np.average(embedings, axis=0)

        if self.page_averages is None:
            self.page_averages = average
        else:
            np.stack((self.page_averages, average))

        a = tf.constant(average, dtype=tf.float64)

        distance_indx = None

        for i, embedded in enumerate(embedings):

            b = tf.constant(embedded, dtype=tf.float64)

            # Calculating result
            res = tf.math.squared_difference(a, b)

            thisDist = tf.reduce_sum(res)

            this_dist_indx = np.array([(thisDist, i)], dtype=[('dist', '<f8'), ('index', '<i4')])

            if i == 0:
                distance_indx = this_dist_indx
            else:
                distance_indx = np.append(distance_indx, this_dist_indx, axis=0)

        distance_indx = np.sort(distance_indx, order=['dist'])

        #

        if self.dist_indxs is None:
            self.dist_indxs = distance_indx
        else:
            self.dist_indxs = np.append(self.dist_indxs, distance_indx)

        return distance_indx

    def embedings(self, sentences, debuging=False):
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        # Print the embeddings
        if debuging:
            for sentence, embedding in zip(sentences, embeddings):
                print("Sentence:", sentence)
                print("Embedding:", embedding)
                print("")

        return embeddings

    def train_AVE(self):

        self.AVE = ave.AVE();

        dataset = None

        for i, data in enumerate(self.page_data):
            if i != 0:

                dataset = np.append(dataset, data.embedings, axis=0)
            else:
                dataset = data.embedings

        self.AVE.set_data(dataset)
        self.AVE.create_model(3)
        self.AVE.epochs = 10
        self.AVE.train()
        self.AVE.test(1)

    def highlight_with_AVE(self):

        for page, data in zip(self.doc, self.page_data):

            for sentence, embed in zip(data.sentences, data.embedings):

                r, g, b = self.AVE.encode_embed_01(embed)


                text_instances = page.search_for(sentence)

                if text_instances is None:
                    continue

                ### HIGHLIGHT
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=[r, g, b])
                    highlight.update()



# materially discursive practice is collaborating with something to find an understanding of it.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Hello you!")
    test_doc = Doc()
    print("Loading Doc")
    test_doc.loadpdf("xcoax paper.pdf")#a-matter-of-difference-karen-barad-ontology-and-archaeological-bodies.pdf")
    print("Finding the embedded data")
    test_doc.get_embed_data()
    print("Highlighting divergent")
    #test_doc.highlight_divergent()
    # print("Reading highlighted")
    #test_doc.read_highlights()
    test_doc.train_AVE()
    test_doc.highlight_with_AVE()
    print("Saving!")
    test_doc.save("tst")
