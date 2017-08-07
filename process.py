"""
Processes images from the PPPL Calibration Framework
"""

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


def read_imgs(path, foldernumber):
    folderpath = os.path.join(path, str(foldernumber))

    backgroundpath = os.path.join(folderpath, "background.png")

    background = cv2.imread(backgroundpath)[80:330, 170:500]
    imgs = []

    for fn in sorted(os.listdir(folderpath)):
        if fn != 'background.png' and fn != 'data.csv':
            path = os.path.join(folderpath, fn)
            img = cv2.imread(path)[80:330, 170:500]
            imgs.append(img)

    return (imgs, background)


def process_power(imgs, background):
    max_power = 0
    for img in imgs:
        diffimg = cv2.absdiff(img, background)
        cv2.imshow('diff', diffimg)
        power = np.amax(diffimg)
        if power > max_power:
            max_power = power
        cv2.waitKey(1)

    return max_power


def process_frequency(imgs, background):
    return 0


def process_location(imgs, background):
    return (0, 0)


def process_scan_imgs(path):
    metadata = pd.read_csv("{}/data.csv".format(path))
    data = pd.DataFrame().reindex_like(metadata)

    data.loc[:, 'Time'] = metadata.loc[:, 'Time']
    data.loc[:, 'Folder Number'] = metadata.loc[:, 'Folder Number']

    print(metadata)
    print(data)

    for i, val in enumerate(data['Folder Number']):
        print("Processing:", i)
        imgs, background = read_imgs(path, val)

        data.loc[i, 'Power'] = process_power(imgs, background)
        data.loc[i, 'Frequency'] = process_frequency(imgs, background)
        data.loc[i, 'X Axis'], data.loc[i,
                                        'Y Axis'] = process_location(imgs, background)

        del imgs
        del background

    cv2.destroyAllWindows()
    print(data)

    return metadata, data


def organize_data(metadata, data):
    # powerdata = pd.DataFrame(columns=pd.unique(
    #    metadata['X Axis']), index=pd.unique(metadata['Y Axis']))

    power_coords = pd.unique(metadata.loc[:, 'Power'])
    y_coords = pd.unique(metadata.loc[:, 'Y Axis'])
    x_coords = pd.unique(metadata.loc[:, 'X Axis'])

    powerdata = xr.DataArray(
        data=np.zeros(shape=(
            len(power_coords),
            len(y_coords),
            len(x_coords),
        )),
        coords=[
            power_coords,
            y_coords,
            x_coords,
        ],
        dims=(
            "power",
            "yaxis",
            "xaxis",
        )
    )

    print(powerdata)

    for i, val in enumerate(data['Power']):
        if metadata.loc[i, 'Frequency'] == 0.5:
            print(metadata.loc[i])
            x = metadata.loc[i, 'X Axis']
            y = metadata.loc[i, 'Y Axis']
            p = metadata.loc[i, 'Power']

            powerdata.loc[dict(
                xaxis=x,
                yaxis=y,
                power=p
            )] = data.loc[i, 'Power']

    #powerdata = powerdata.convert_objects(convert_numeric=True)
    print(powerdata)

    return powerdata


def power_slopes(data):
    slopedata = xr.DataArray(
        data=np.zeros(shape=(
            len(data.yaxis.values),
            len(data.xaxis.values),
        )),
        coords=[
            data.yaxis.values,
            data.xaxis.values,
        ],
        dims=(
            "yaxis",
            "xaxis",
        )
    )
    print(slopedata)

    for x in data.xaxis:
        for y in data.yaxis:
            powers = data.loc[dict(xaxis=x, yaxis=y)]
            print('powers.values', powers.values)
            print('powers.power.values', powers.power.values)
            coef = np.polyfit(y=powers.values, x=powers.power.values, deg=1)

            slopedata.loc[dict(
                xaxis=x,
                yaxis=y,
            )] = coef[0]

            powers.plot()
    plt.show()

    
    print(slopedata)
    return slopedata


def plot_data(data):
    # plot heatmap
    ax = sns.heatmap(data)

    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    # save figure
    #plt.savefig('seabornPandas.png', dpi=100)
    plt.show()


if __name__ == '__main__':
    #meta, data = process_scan_imgs("realdeal2/RealDeal2")
    #meta.to_csv('realdeal2/meta_save.csv')
    #data.to_csv('realdeal2/data_save.csv')
    meta = pd.read_csv('realdeal2/meta_save.csv')
    data = pd.read_csv('realdeal2/data_save.csv')
    powerdata = organize_data(meta, data)
    slopedata = power_slopes(powerdata)
    plot_data(slopedata)
    import IPython
    IPython.embed()
    # for p in power.loc[:, :]:
    #    plot_data(p)
