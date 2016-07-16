from Reader import Reader
from DigitOCR import DigitOCR


ocm = DigitOCR()

ocm.create_training_data()
ocm.train()


reader = Reader()



reader.load_image("images/sudoku6.jpg")
reader.clean_image()
reader.rectify_perspective()
reader.show_rectified()
reader.cut_image_from_clean()


reader.margin_cases()


mat = reader.convert_to_matrix(ocm)
print mat


