### I am writing this code to make my emulator user friendly, So using this code user can give a range of feature value and then it will ask for how many results do \n
###  you want from this. As you will give it to your choice it will take it as input and will produce output. In short this is a code which will ask for your choice \n
###  and will generate result. you can exit also whenever you want.



### Libraries and model is same as in Emulater.py file

def get_plot(index,xh_val,no_plot):
    
    k = 0
    for i in index:
             
        k = k + 1
        a, b = y_test.loc[i] , y_pd_df.loc[i]

        error_ = abs(a-b)
        rel_error = (error_/a)*100


        y_max1 = 1 + max(rel_error)
        y_min = min(rel_error) - 0.1

        plt.figure(figsize=(20, 7))
        G = gridspec.GridSpec(1, 3,wspace=0.3,hspace=0.5)
        ax1 = plt.subplot(G[0, :2])
   
        ax3 = plt.subplot(G[0, 2])

        plt.suptitle(r"Figure %d :- Power Spectrum at $x_h$ = %.3f" %(k,xh_val),
                     fontsize = 22,color = "firebrick",y = 1.1)

        ax1.loglog(K_vals,b,'go--',label = "Predicted PS")
        ax1.loglog(K_vals,a,'ko',label = "Actual PS")
        ax1.set_title("Power Spectrum",color = 'blue',fontsize = 17)
        ax1.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax1.set_ylabel(r" $\Delta ^{2} (mK^2)$ $ \longrightarrow$" ,fontsize = 17,color = 'darkgreen')
        ax1.legend(fontsize = 15)
        p,q,r = md_data.drop(["X_h"],axis = 1).iloc[i]
        ax1.set_title(r'Parameters:  $M_{min}$ = %.3f , $N_{ion}$ = %.3f , $R_{mfp}$ = %.3f'%(p,q,r),
                 fontsize=17,color='purple',bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 10},loc='right',x= 1.07,y = 1.134)

        ax3.set_box_aspect(11/12)
        ax3.semilogx(K_vals,rel_error,'go--',label = " Relative error in %")

    
        ax1.yaxis.set_major_formatter('{x:.0f}')

        ax3.set_ylabel(r"Error $ \longrightarrow$" ,fontsize = 15,color = 'darkgreen')
        ax3.set_title("Relative Error in Prediction",color = 'blue',fontsize = 17)
        ax3.set_xlabel(r"k (Mp$c^{-1}$) $ \longrightarrow$",fontsize = 17,color = 'darkgreen')
        ax3.legend(fontsize = 12)
        plt.show()
        if k == no_plot:
            break





def interaction():
        
    s = 0
    while s<1:
        print("")
        print("[bold bright_red] Instructions:-")
        print("")
        print("[bright_blue] Please provide me the value of neutral fraction at which you want to see the Power Spectrum.")
        print("[bright_blue italic underline] Note: The value of neutral fraction should be between 0 to 1.")
        print("")
        xh_val = input("Enter a number (or 'q' to quit): ")
        if xh_val.lower() == 'q':
            print("[bold bright_magenta] Bye! We are leaving the loop.")
            break
        else:
            xh_val = float(xh_val)
            if 0<=xh_val<1:

                xx = xh_test.loc[xh_test["X_h"]==xh_val]

                if len(xx) ==0:

                    print(" ")
                    print("[green] ===>  Sorry! I didn't found any Power Spectrum plot at neutral fraction %.3f in testing data set."%(xh_val))
                    print(" ")
                    print("[green] ===>  Provide a range so that I can give you neighbour neutral fraction values around this %.3f value."%xh_val)
                    print("")
                    print("[green] ===>  If you want very close value then please provide very short range like 0.01,0.011 etc.")
                    rang = input("Enter the range (or 'q' to quit): ")
                    if rang.lower() == 'q':
                                 print("[bold bright_magenta] Bye! We are leaving the loop.")
                                 break
                    else:
                        ss = 0
                        while ss<1:
                            
                            rang = float(rang)
                            xh1 =  xh_val - rang
                            xh2 = xh_val + rang
                            new_index = xh_test.query(" %.3f <= X_h <= %.3f"%(xh1,xh2)).index.tolist()
                            NF_array = xh_test.loc[new_index]
                            NF_array = NF_array['X_h'].to_numpy()
                            if len(NF_array)==0:
                                print("[bright_blue] Sorry! there is no any neighbour in this range. Please increase your range.")
                                rang = float(input("Enter new range:  "))

                            else:
                                print()
                                print("[green] ===>  oh! I found {} neutral fraction values which are neighbours of previous value".format(len(new_index)))
                                print(" ")
                                print("[green] ===>  List of these values is here:")
                                print()
                                print(np.sort(NF_array))
                                print()
                                print("[green] ===>  Please select one of the neutral fraction value from above list at which you want to see the power spectrum.")
                                print(" ")
                                print(" ")
                                ss = ss+1

                else:
                    if len(xx)==1:
                        s = s+1
                        print("[green] ===>  I found {a} Power Spectrum plots which is here:-".format(a = len(xx)))
                        index = xx.index
                        get_plot(index,xh_val,len(xx))
                    else:
                        s = s+1
                        print("[green] ===>  I found {a} Power Spectrum plots. How many do you want to see?. Please provide me a number.".format(a = len(xx)))
                        no_plot = input("Enter a number (or 'q' to quit): ")
                        if no_plot.lower()=='q':
                            print("[bold bright_magenta] Bye! We are leaving the loop.")
                            break
                        else:
                            no_plot = float(no_plot)

                            if no_plot<1:


                                print("[green] ===>  oh! I think you made a mistake[/green], [bold cyan] Please provide me a valid number[/bold cyan].")
                                print("[green] ===>  If this is not your mistake then please enter 1 or enter 0 if you want to change number of plot.")
                                choice = float(input("Enter your choice:  "))
                                san = 0
                                while san<1:

                                    if choice==1:
                                        print("[bold cyan italic] ===>  Better luck next time")
                                        san = san +1
                                    elif choice==0:
                                        no_plot = input("Enter a number howmany plots do you want (or 'q' to quit): ")
                                        if no_plot.lower() == 'q':
                                            print("[bold bright_magenta] Bye! We are leaving the loop.")
                                            break
                                        else:
                                            no_plot = float(no_plot)

    #                                         print("[bold cyan italic] ===>  I got your input that you want to see {} plots. If you want to continue with {} plots then type 0 or if you want to change number of plots then type 1".format(no_plot,no_plot))
    #                                         feed = int(input("Enter your choice:  "))
    #                                         if feed ==1:
    #                                             print("[bold cyan italic] ===>  Please re-enter howmany plots do you want.")
    #                                             no_plot = int(input("Enter a number:  "))
    #                                             index = xx.index
    #                                             get_plot(index,xh_val,no_plot)

    #                                         else:
                                            index = xx.index
                                            get_plot(index,xh_val,no_plot)
                                            san = san+1
                                    else:
                                        print("[bold bright_magenta] You are giving invalid input. Try again!")
                                        choice = float(input(" Enter your choice again  "))

                            else:
  
                                index = xx.index
                                get_plot(index,xh_val,no_plot)
            else:
                print("[bright_red italic underline dim] Stop! You are giving an invalid input. Please read instructions again.")
    
    print("")

    
    
def Interactive_result():
    
    while True:
        interaction()
        print("")
        print("[bold magenta] ===> Do you want to continue it or quit it? If you want to continue then enter [underline]'c'[/underline] or if you want to quit it then enter [underline]'q'[/underline]")
        print("")
        sk = (input("Enter your choice: "))
        if sk.lower() == 'q':
            break