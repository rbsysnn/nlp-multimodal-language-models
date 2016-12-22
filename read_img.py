import os 
import urllib                                                                             
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#list of lists - num_of_im the total number of images


def show_images(url,generated_captions,real_captions,bleu_score,num_of_im=4):
    
    
    #for 4 images
    num_of_images=0
    
    #create plots
    f, ax = plt.subplots(2, num_of_im/2,figsize=(10,10))
    ax=ax.ravel()
    
    
    #check for all urls
    for url_id in range(len(url)):
        #while images <4

        if num_of_images<num_of_im:
            
            #print ''.join(url[j])
            #check if image is in mscoco
            if "mscoco" in ''.join(url[url_id]):
                    
                #img_id
                img_id= ''.join(url[url_id]).rpartition('/')[2]
                url[url_id]=["http://mscoco.org/images/"+img_id]
                urllib.urlretrieve(''.join(url[url_id]), img_id)
                try:
                    img=mpimg.imread(img_id)
                except IOError :
                    # print("Not a valid image")
                    continue
                ax[num_of_images].imshow(img)
                ax[num_of_images].set_xticks([])
                ax[num_of_images].set_yticks([])
                plt.setp(ax[num_of_images].get_yticklabels(), visible=False)
                plt.setp(ax[num_of_images].get_xticklabels(), visible=False)
                ax[num_of_images].set_xlabel("Generated caption : "+''.join(generated_captions[url_id])+
                "\n"+"Real caption : "+''.join(real_captions[url_id]))
                
                
                # os.remove(img_id)
                num_of_images+=1

        #else break        
        else:
            break
    
    #show plots         
    plt.tight_layout()
    plt.show()




#url something like this 
url=[["datasets/mscoco/images/afadga/afagagda/agaga/agagda/0000000171678"],["/mscoco/asfafa"],["mscoco/107959"]]

# captions like this
generated_captions=[["1"],["2"],["3"],["4"]]
real_captions=[["1"],["3"],["2"],["3"],["4"]]
bleu_score = [

show_images(url,generated_captions,real_captions)



