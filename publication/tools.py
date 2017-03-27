from second_hand_house.toolbox import *


def build_volume_dataset():
    fw = open('artificial_volumes.txt', 'w+')
    for i in range(1, 100):
        print(i)
        fw.write(str(i) + '\n')
        for j in range(1, 50):
            print(str(i)+'('+str(j)+')')
            fw.write(str(i)+'('+str(j)+')'+'\n')
    fw.close()


def build_volume_dataset4train():
    fw = open('temp_artificial_volumes.txt', 'w+')
    for i in range(1, 100, 2):
        print(i)
        fw.write(str(i) + '\n')
        for j in range(1, 50, 2):
            print(str(i)+'('+str(j)+')')
            fw.write(str(i)+'('+str(j)+')'+'\n')
    fw.close()


def load_years(fp):
    years_l = []
    for line in open(fp, 'r'):
        str_year = sample_pretreatment_disperse_number2(line).strip()
        # print(str_year)
        years_l.append(str_year)
    years = [year.split() for year in years_l]
    return years


def load_volume(fp):
    volume_l = []
    for line in open(fp, 'r'):
        str_volume = sample_pretreatment_disperse_number2(line).strip()
        volume_l.append(str_volume)
    volumes = [volume.split() for volume in volume_l]
    return volumes


def load_pages(fp):
    pages_l = []
    for line in open(fp, 'r'):
        str_pages = sample_pretreatment_disperse_number2(line).strip()
        pages_l.append(str_pages)
    pages_l = [pages.split() for pages in pages_l]
    return pages_l


def load_all_attribute_data():
    journals = load_all_journals('../v3/all_journal_1614_.txt')
    titles = load_all_v3titles('../v3/all_title_1517347_.txt')
    authors = load_all_v3authors('../v3/all_author_1137677_.txt')
    years = load_years('../dataset_workshop/temp_year.txt')
    volumes = load_volume('../dataset_workshop/artificial_volumes.txt')
    pages = load_pages('../dataset_workshop/temp_page.txt')

    x_text = titles + authors + journals + years + volumes + pages

    return x_text


def build_complete_vocab():
    print('vocab')
    all_samples = load_all_attribute_data()
    vocab = makeWordList(all_samples)
    save_dict(vocab, 'publication_complete_dict.pickle')


def build_y_train_publication_all_attribute(titles, authors, journals, years, volumes, pages):
    # print("Building label dict:")
    title_labels = [0 for i in range(len(titles))]
    author_labels = [1 for i in range(len(authors))]
    journal_labels = [2 for i in range(len(journals))]
    year_labels = [3 for i in range(len(years))]
    volume_labels = [4 for i in range(len(volumes))]
    page_labels = [5 for i in range(len(pages))]

    y_t = title_labels + author_labels + journal_labels + year_labels + volume_labels + page_labels
    label_dict_size = 6

    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    # print("Preparing y_train over!")
    return y_train, label_dict_size

if __name__ == '__main__':
    # build_volume_dataset4train()
    # for i in range(0, 10, 2):
    #     print(i)
    # years = load_years('../dataset_workshop/temp_year.txt')
    # print(years)
    #
    # volumes = load_volume('temp_artificial_volumes.txt')
    # print(volumes)
    #
    # pages = load_pages('../dataset_workshop/temp_page.txt_kb')
    # print(pages)
    build_complete_vocab()