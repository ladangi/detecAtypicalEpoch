# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:34:40 2019

@author: Laura Gil
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import LinearFIR

def loadData(name_data):
    """ load the signal data from a .txt file
    
    Parameters:
        
        name_data: file name
        
    Returns:
        matrix_chanels: numpy.ndarray
                        matrix with the signal channel data
    """
    
    senalEEG=[]
    with open(name_data) as senal_str:
        for line in senal_str:
            if line[0] != '%':
                cut = line.split(',')
                data = cut[1:-1]
                senalEEG.append(data)
    senal_data = pd.DataFrame(senalEEG)
    matrix_chanels = np.array(senal_data, dtype=float)
    return matrix_chanels


def sectionEpochs(chanels, duration_epochs, fs):
    """section the channels by epochs
    
    Parameters:
        chanels: numpy.ndarray
                 matrix with the signal channel data
                 
        duration_epochs: type in
                        time that lasts one epoch
                        
        fs: type in
            sampling frequency
                 
    Returns:
        matrix_section: numpy.ndarray
                        matrix that contains the channels sectioned by epochs
    
    """
    matrix_channels=chanels.copy()
    row,column = matrix_channels.shape
    
    total_time = int(row/fs)#total time of the signal 
    num_data = int((duration_epochs*row)/total_time)#number of data that each epoch contains
    num_epochs = int(row/num_data)#number of epoch per channel
    total_row = num_data*num_epochs# new number of seasons per channel
    matrix_channels = matrix_channels[:total_row,:]#the rows of the matrix are trimmed until you have the new number of rows

    row,column = matrix_channels.shape
    matrix_transpose = np.transpose(matrix_channels)
    matrix_section = matrix_transpose.reshape((column, num_epochs, num_data)) #the matrix is organized for epochs
    return matrix_section

def epochsFilter(chanels):
    """filters the signal and eliminates the offset value
    
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
    Returns:
        matrix_filter: numpy.ndarray
                        Matrix that contains the filtered channels sectioned by epochs
    """
    matrix_filter = chanels.copy()
    row,column,depth = matrix_filter.shape        
    matrix_filter = matrix_filter.reshape((row, column*depth))   
    lowpass = 50
    higpass = 1
    
    for index_canal in range(row):
        channel = matrix_filter[index_canal]
        EEG = LinearFIR.eegfiltnew(channel, 250, higpass, 0, 0)
        EEG = LinearFIR.eegfiltnew(EEG, 250, 0, lowpass, 0)
        matrix_filter[index_canal] = EEG 
        
    matrix_filter = matrix_filter.reshape((row, column, depth))
    
    return matrix_filter



def extremValue(chanels,threshold):
    """ Extreme value method, eliminates the time of all channels that exceed the threshold
    
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        threshold: type int
        
    Returns:
        matrix_extremValue: numpy.ndarray
                            Matrix containing epoch that did not exceed the threshold
                            
        epoch_delete: numpy.ndarray
                      It contains the deleted epoch
                
    """
    matrix_extremValue=chanels.copy()
    row,column,depth = matrix_extremValue.shape
    epoch_delete = np.array([])
    mask=np.absolute(matrix_extremValue) >= threshold
    for index_column in range(column):
        if np.any(mask[:,index_column,:]):
            if not index_column in epoch_delete:
                epoch_delete = np.append(epoch_delete, index_column)
    matrix_extremValue = np.delete(matrix_extremValue, epoch_delete, 1)
    for removed in epoch_delete:
        print('se eliminó la epoca: '+str(removed)+' Por el método de valores extremos')
    return matrix_extremValue, epoch_delete   

def linearRegression(chanels, fs):
    """Calculates the linear regression of each epoch, 
        if it exceeds the threshold, the value of the slope is subtracted
      
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs                 
       
    Returns:
        matrix_reg: numpy.ndarray
                    matrix without atypical epochs
        
    """
    matrix_reg = chanels.copy()
    row,column,depth = matrix_reg.shape
    for indexRow in range(row):
        for indexColumn in range(column):
            y = matrix_reg[indexRow][indexColumn]
            time = np.arange(0, len(y)/fs, 1/fs)
            data_reg = stats.linregress(time,y)
            if data_reg[0] >= 0:
                for data in range(0,len(y)):
                    t=data/fs
                    pending = data_reg[0]*t + data_reg[1]                    
                    matrix_reg[indexRow][indexColumn][data] = matrix_reg[indexRow][indexColumn][data] - pending
    return matrix_reg

def Kutorsis(chanels, thresholdlow, thresholdhig):
    """If the distribution of the data have a tendency 
        to a normal curve, the epoch of all the channels is eliminated
     
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        threshold: type int
        
    Returns:
        matrix_kurt: numpy.ndarray
                    matrix with the times that have a normal data distribution
        
        epochs_delete: numpy.ndarray
                    matrix with the times that do not have a normal data distribution
        
    """
    matrix_kurt = chanels.copy()
    row,column,depth = matrix_kurt.shape
    epochs_delete = np.array([])
    
    for indexRow in range(row):
        for indexColumn in range(column):
            epoch = matrix_kurt[indexRow,indexColumn,:]
            valor_kutursis = stats.kurtosis(epoch, fisher = True)
            if np.any(valor_kutursis > thresholdlow and valor_kutursis < thresholdhig):
                if not indexColumn in epochs_delete:
                    epochs_delete = np.append(epochs_delete, indexColumn)
    matrix_kurt = np.delete(matrix_kurt, epochs_delete, axis=1)
    for removed in epochs_delete:
        print('se eliminó la epoca: '+str(removed)+ 'Por curtosis')
    
    return matrix_kurt, epochs_delete

def Spectrum(channel,threshold,fs):
    """analyzes the spectrum of each channel and if this 
    exceeds the threshold the epoch of all channels is eliminated
     
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        threshold: type int
        
    Returns:
        matrix_kurt: numpy.ndarray
                    matrix with the values with which they do not exceed the threshold
        
        epochs_delete: numpy.ndarray
                   list with the times that were eliminated
        
    """
    
    
    matrixSpectrum = channel.copy()
    row,column,depth = matrixSpectrum.shape
    matrixSpectrum = matrixSpectrum.reshape((row, column*depth))
    epochs_delete = np.array([])
    
    for indexRow in range(row):
        fChannel,PxxChannel = signal.welch(matrixSpectrum[indexRow], fs, nperseg=1024)
        averageChannel = np.mean(PxxChannel)
        matrixSpectrum = matrixSpectrum.reshape((row, column, depth))
        for indexColumn in range(column):
            fEpoch,PxxEpoch = signal.welch(matrixSpectrum[indexRow, indexColumn,:], 250, nperseg=1024)
            
            if np.any((PxxEpoch - averageChannel) > threshold):
                if not indexColumn in epochs_delete:
                    epochs_delete = np.append(epochs_delete, indexColumn)
    matrixSpectrum = np.delete(matrixSpectrum, epochs_delete, axis=1)
    print(matrixSpectrum.shape)
    for removed in epochs_delete:
        print('se eliminó la epoca: '+str(removed)+ 'Por Spectrum')
    
    return matrixSpectrum, epochs_delete 


def graphChanels(chanels,fs, list_chanels='Todos'):
    """Graph one or all channels
    
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        fs: type int
            sampling frequency
        
        list_chanels: numpy.ndarray
                        Save the channels you want to graph
    """
    row,column,depth = chanels.shape    
    if len(list_chanels) == 1:
        plt.figure()
        l = 0
        EEG_chanel = np.array([])
        for indexColumn in range(column):
            EEG_chanel = np.concatenate((EEG_chanel, chanels[list_chanels[0],indexColumn,:])) # concatenates the epochs of the channel indexColumn 
               
        l = len(EEG_chanel)
        time=np.arange(0,l/fs,1/fs)  
        plt.plot(time,EEG_chanel)
        plt.grid()
        plt.xlabel('Tiempo [s]',fontsize = 7)
        plt.ylabel('Amplitud [uV]',fontsize = 7)
        plt.title('canal '+str(list_chanels[0]),fontsize = 7)
        plt.show()        
    
    else:
        plt.figure(figsize = (20,60)) 
        plt.subplots_adjust(bottom=0.086, top=0.951, right=0.981, left=0.061, hspace=1.0,wspace=0.157)
    
        if list_chanels == 'Todos':
            list_chanels = np.arange(row)
            
        list_chanels = np.sort(list_chanels) 
        z = 1 
        for channel in list_chanels:
            l=0
            EEG_chanel = np.array([])
            for indexColumn in range(column):
                EEG_chanel = np.concatenate((EEG_chanel, chanels[channel,indexColumn,:])) 
            
            l = len(EEG_chanel)
            time=np.arange(0,l/fs,1/fs)  
            plt.subplot(6,2,z)            
            plt.plot(time,EEG_chanel)
            plt.grid()
            if z == 5:
                plt.ylabel('Amplitud [uV]',fontsize = 9)
                
            if z==6:
                plt.ylabel('Amplitud [uV]',fontsize = 9)
            
            if z >= 10:
                plt.xlabel('Tiempo [s]')
            #plt.xlim(0,205)
            plt.title('Canal '+str(channel),fontsize = 9)
            plt.show()
            z=z+1
    return True


def graphEpochsDelet(chanels, epochs_delet, fs, list_chanels='Todos'):
    """Graph one or all the seasons eliminated by the detection methods of atypical times
   
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
         
        epochs_delet: numpy.ndarray
                      list with the eliminated epoch that the user wants to graph
        fs: type int
            sampling frequency
        
        list_chanels: numpy.ndarray
                        Save the channels you want to graph
    
    """
    row,column,depth = chanels.shape
    
    if len(list_chanels) == 1:
        plt.figure()
        l=0
        eeg_canal = np.array([])
        for indexColumn in range(column):
            if indexColumn in epochs_delet:
                eeg_canal = np.concatenate((eeg_canal, chanels[list_chanels[0],indexColumn,:])) 
            else:
                eeg_canal = np.concatenate((eeg_canal, np.zeros(depth)))
        l = len(eeg_canal)
        time=np.arange(0,l/fs,1/fs)  
        plt.plot(time,eeg_canal)
        plt.grid()
        plt.xlabel('Tiempo [s]',fontsize = 9)
        plt.ylabel('Amplitud [uV]',fontsize = 9)
        plt.title('Canal '+str(list_chanels[0]),fontsize = 9)
        plt.show()        
    
    else:
        plt.figure(figsize = (20,60)) #figsize = (10,50)
        plt.subplots_adjust(bottom=0.08, top=0.95, right=0.975, left=0.065, hspace=0.965,wspace=0.225)
        epochs_delet = np.sort(epochs_delet)
        if list_chanels == 'Todos':
            list_chanels = np.arange(row)
            
        list_chanels = np.sort(list_chanels) 
        
        z = 1
        for indexChanel in list_chanels:
            l=0
            eeg_canal = np.array([])
            for indexEpoch in range(column):
                if indexEpoch in epochs_delet:
                    eeg_canal = np.concatenate((eeg_canal, chanels[indexChanel,indexEpoch,:])) 
                else:
                    eeg_canal = np.concatenate((eeg_canal, np.zeros(depth)))
            l = len(eeg_canal)
            time=np.arange(0,l/fs,1/fs)  
            plt.subplot(6,2,z)           
            plt.plot(time,eeg_canal)
            plt.grid()
            
            if z == 5:
                plt.ylabel('Amplitud [uV]',fontsize = 9)
                
            if z==6:
                plt.ylabel('Amplitud [uV]',fontsize = 9)   
            
            if z >= 10:
                plt.xlabel('Tiempo [s]')
                
            plt.xlim(0,205)            
            plt.title('Canal '+str(indexChanel),fontsize = 9)        
            plt.show()
            z=z+1
    return True


def graphEEG(chanels,fs):
    """graph all the channels with the filtered data differentiating the epoch
    Parameters:
        chanels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
         fs: type int
             sampling frequency    
    """
    row,column,depth = chanels.shape
    plt.figure() 
    plt.subplots_adjust(bottom=0.09, top=0.93, right=0.98, left=0.065, hspace=0.2,wspace=0.2)
    T = depth/fs
    for indexRow in np.arange(row):
        l=0
        eeg_canal = np.array([])
        for indexColumn in range(column):
            eeg_canal = np.concatenate((eeg_canal, chanels[indexRow,indexColumn,:]))            
        eeg_canal = eeg_canal + 250*indexRow
        l = len(eeg_canal)
        time=np.arange(0,l/fs,1/fs)
        x_axis = np.linspace(0,column*T,column)
        xrotulo=np.arange(0,len(x_axis))
        y_axis = np.linspace(0,row*227,row)
        yrotulo=np.arange(0,len(y_axis))
        plt.plot(time,eeg_canal)        
        plt.xlabel('Epocas')
        plt.ylabel('Canal')
        plt.title('Canales')
        plt.yticks(y_axis,yrotulo)
        plt.xticks(x_axis,xrotulo)
        plt.xlim(0,205)        
        plt.grid(True)        
        plt.show()  
    return True


def graphONEepoch(channels,channel, epoch, fs):
    """graph one time
     
     Parameters:
        channels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        channel: type int
                 channel to which the epoch that you want to graph belongs
        
        epoch: type int
                Epoch you want to graph
                 
         fs: type int
             sampling frequency  
    """
    
    row,column,depth = channels.shape      
    plt.figure()
    eeg_epoca = channels[channel,epoch,:]
    lengthEpoch = len(eeg_epoca)
    time=np.arange(0,lengthEpoch/fs,1/fs)  
    plt.plot(time,eeg_epoca)
    plt.grid()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [uV]')
    plt.title('Canal '+str(channel)+'-epoca '+str(epoch))
    plt.show()
    return True

def graphSpectrum(channels, channel, fs):
    """Graph the power spectrum of a channel
    
    Parameters:
        channels: numpy.ndarray
                 matrix that returns the "Spectrum" function
                 
        channel: type int
                 channel that you want to graph 
               
         fs: type int
             sampling frequency  
    
    """
    ChannelSpectrum = channels.copy()  
    row,column,depth = ChannelSpectrum.shape
    ChannelSpectrum = ChannelSpectrum.reshape((row, column*depth))
    EEGspectrum = ChannelSpectrum[channel]
    fChannel,PxxChannel = signal.welch(EEGspectrum, fs, nperseg=1024)
    #plt.plot(fChannel,PxxChannel)
    plt.semilogy(fChannel,PxxChannel)#Grafica del espectro
    plt.grid()
    plt.ylabel('Densidad de Potencia [uV^2/Hz]')
    plt.xlabel('Frecuencia [Hz]')
    plt.title('Periodograma del canal' + str (channel))
    plt.show()
    return True

def GraphCompareEpochs(channels,channel, list_epochs, fs):
    """comparative graph between six epochs
    
    Parameters:
        channels: numpy.ndarray
                 matrix that contains the channels sectioned by epochs
                 
        channel: type int
                 channel to which the epoch that you want to graph belongs
        
        list_epochs: numpy.ndarray
                     Epochs you want to graph
                 
         fs: type int
             sampling frequency  
    
    """
    row,column,depth = channels.shape
    list_epochs = np.sort(list_epochs)
    
    num_windows = int(len(list_epochs)/6)
    
    if not len(list_epochs)%6 == 0:
        num_windows = num_windows + 1
        
    for window in range(num_windows):
        plt.figure(figsize = (10,40))
        z = 1 
        start = window*6
        end = (window+1)*6
        for epoch in range(start,end):
            eeg_epoca = channels[channel,int(list_epochs[epoch]),:]
            lengthEpoch = len(eeg_epoca)
            time=np.arange(0,lengthEpoch/fs,1/fs)  
            plt.subplot(3,2,z)
            plt.plot(time,eeg_epoca)
            plt.grid()
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud [uV]')
            plt.title('epoca '+str(list_epochs[epoch]))
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
            plt.show()
            z=z+1
    return True


def GraphCompareMethods(matrix_original, matrix1, matrix2, matrix3, matrix4, matrix5, channel, fs):
    """It makes a comparison between the applied methods
    
    Parameters:
        matrix_original: numpy.ndarray
                 Matrix that returns the "sectionEpochs" function
        
        matrix1: numpy.ndarray
                Matrix that returns the "epochsFilter" function
                
        matrix2: numpy.ndarray
                    Matrix that returns the "extremValue" function

        matrix3: numpy.ndarray
                    Matrix that returns the "linearRegression" function                         

        matrix4: numpy.ndarray
                    Matrix that returns the "Kutorsis" function        

        matrix5: numpy.ndarray
                    Matrix that returns the "Spectrum" function 
                    
        channel: type int
                 channel to which the epoch that you want to graph belongs
                 
         fs: type int
             sampling frequency    
    """    
    
    row,column,w = matrix_original.shape
    row1,column1,w1 = matrix1.shape
    row2,column2,w2 = matrix2.shape
    row3,column3,w3 = matrix3.shape
    row4, column4,w4 = matrix4.shape
    row5, column5,w5 = matrix5.shape
   
    
    y = np.array([column, column1, column2, column3, column4, column5])
    plt.figure()
    plt.subplots_adjust(top=0.915,bottom=0.09, left=0.08, right=0.975,hspace=0.48, wspace=0.195)
    z = 1 
    for matrix in range(6):
        quantityEpoch = y[matrix] # number of epochs of the matrix
        eeg_canal = np.array([])
        for epoch in range(quantityEpoch):
            if matrix == 0:
                eeg_canal = np.concatenate((eeg_canal, matrix_original[channel[0],epoch,:]))
                
            if matrix == 1:
                eeg_canal = np.concatenate((eeg_canal, matrix1[channel[0],epoch,:]))
                
            if matrix == 2:
                eeg_canal = np.concatenate((eeg_canal, matrix2[channel[0],epoch,:]))
                
            if matrix == 3:
                eeg_canal = np.concatenate((eeg_canal, matrix3[channel[0],epoch,:]))
                
            if matrix == 4:
                eeg_canal = np.concatenate((eeg_canal, matrix4[channel[0],epoch,:]))
                
            if matrix == 5:
                eeg_canal = np.concatenate((eeg_canal, matrix5[channel[0],epoch,:]))
                
            
        lengthChannel = len(eeg_canal)
        time=np.arange(0,lengthChannel/fs,1/fs) 
        plt.subplot(3,2,z)
        plt.grid()
      
        if z == 1:
            plt.title('EEG original')
            
        if z == 2:
            plt.title('EEG filtrada')
            
        if z == 3:
            plt.ylabel('Amplitud [uV]')
            plt.title('Metodo-Extrem values')
            
        if z == 4:
            plt.ylabel('Amplitud [uV]')
            plt.title('Metodo-Regresión Linal')
            
        if z == 5:
            plt.xlabel('Tiempo [s]')
            plt.title('Método-Kurtosis')
            
        if z == 6:
            plt.xlabel('Tiempo [s]')
            plt.title('Método-Espectral')        
       
        plt.plot(time,eeg_canal)
        plt.show()
        z=z+1
    
    return True

#canales = loadData('P1_RAWEEG_2018-11-15_Electrobisturí1_3min.txt')
#senal = sectionEpochs(canales,2,250)
#senal2 = epochsFilter(senal)
#senal3 = extremValue(senal2,50)
#senal4 = linearRegression(senal2, 250)
#senal5 = Kutorsis(senal2, -0.1, 0.3)
#senal6 = Spectrum(senal2,80,250)