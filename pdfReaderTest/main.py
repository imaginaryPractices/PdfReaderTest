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

class Doc():

    def __init__(self):
        self.dist_indxs = None
        self.distance_indx = None
        self.page_averages = None
        self.raw_embedings = None
        self.doc = None
        self.path = None
        self._threshold_intersection = 0.9  # if the intersection is large enough.

    def loadpdf(self, path):

        self.path = path

        self.doc = fitz.open(path)



    def highlight_divergent(self, number_of_divergent = 4, shortest_string = 10):

        for i, page in enumerate(self.doc):

            page_txt = page.get_text()

            text = re.sub("https?:\/\/.*[\r\n]*", "", page_txt)
            text = ''.join(i for i in text if not i.isdigit())
            text = text.replace("Yvonne Marshall and Benjamin Alberti", "")
            text = text.replace("A Matter of Difference: Karen Barad, Ontology and Archaeological Bodies", "")
            text = text.replace("Downloaded from", "")

            sentences = text.split(". ")

            sentences = list(filter(lambda i: len(i) > shortest_string, sentences))

            embeds = self.embedings(sentences)

            if self.raw_embedings is None:
                self.raw_embedings = np.array(embeds)
            else:
                np.vstack((self.raw_embedings, np.array(embeds)))

            distance_indx = self.find_distances(embeds)

            for dist, indx in distance_indx[-number_of_divergent:]:
               ### SEARCH
                text = "Sample text"
                text_instances = page.search_for(sentences[indx])

                ### HIGHLIGHT
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)

            highlight.update()

    def save(self, name = "output"):
        ### OUTPUT
        self.doc.save( name+".pdf", garbage=4, deflate=True, clean=True)

    def link_similar(self):

        for i, page in enumerate(self.doc):
            annot = page.annots()

            while True:
                try:
                    this_annot = annot.__next__()
                    print(self._extract_annot(this_annot, page.get_text()))
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

            this_dist_indx = np.array([(thisDist, i)], dtype=[('dist','<f8'), ('index', '<i4')])

            if i == 0:
                distance_indx = this_dist_indx
            else:
                distance_indx = np.append(distance_indx, this_dist_indx, axis=0)



        distance_indx = np.sort(distance_indx, order=['dist'])

        #

        if self.dist_indxs is None:
            self.dist_indxs = distance_indx
        else:
            self.dist_indxs = np.append( self.dist_indxs, distance_indx)


        return distance_indx

    def embedings(self, sentences=None, debuging=False):
        model = SentenceTransformer('all-MiniLM-L6-v2')

        if sentences is None:
            # Our sentences we like to encode
            sentences = ['This framework generates embeddings for each input sentence',
                         'Sentences are passed as a list of string.',
                         'The quick brown fox jumps over the lazy dog.']

        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        # Print the embeddings
        if debuging:
            for sentence, embedding in zip(sentences, embeddings):
                print("Sentence:", sentence)
                print("Embedding:", embedding)
                print("")

        return embeddings


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    test_doc = Doc()
    print(1)
    test_doc.loadpdf( "a-matter-of-difference-karen-barad-ontology-and-archaeological-bodies.pdf")
    print(2)
    test_doc.highlight_divergent()
    print(3)
    #test_doc.link_similar()
    print(4)
    test_doc.save()

