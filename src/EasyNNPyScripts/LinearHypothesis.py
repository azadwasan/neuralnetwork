from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Now import modules from the directory
from PlotHypothesis import PlotHypothesis
from GDPlayground import OptimizeGD
import time

def PlotTestGradientDescentEvaluation2Features(fig):
    ax = fig.add_subplot(111, projection='3d')

    # Define the range of the function to be centered around the scattered points
    x1_coords = [60, 62, 67, 70, 71, 72, 75, 78]
    x2_coords = [22, 25, 24, 20, 15, 14, 14, 11]

    # My gradient descent solution parameters
    thetas = [0.013080267480039371, 3.0559138415398879, -1.6822470943097785]

    # Reference solution parameters
    thetas2 = [-6.867, 3.148, -1.656]

    y_values = [140, 155, 159, 179, 192, 200, 212, 215]

    PlotHypothesis(ax, x1_coords, x2_coords, thetas, y_values, 'green')
    PlotHypothesis(ax, x1_coords, x2_coords, thetas2, y_values)

    #ReferenceSurface = ax.plot_surface(X1, X2, Y2, color='blue', alpha=0.5)

    ## Create proxy artists for the legend
    proxy1 = Patch(facecolor='green', edgecolor='green', alpha = 0.5, label='Gradient Descent Parameters')
    proxy2 = Patch(facecolor='blue', edgecolor='blue', alpha =0.5, label='Manually Calculated Paramaters')

    ## Add a legend to the plot using the proxy artists
    ax.legend(handles=[proxy1, proxy2])

def PlotTestGradientDescentEvaluation2Features2(fig):
    ax = fig.add_subplot(111, projection='3d')

    # Define the range of the function to be centered around the scattered points
    x1_coords = [4.39930708e-01,  1.22525657e+00, -3.36511220e-01, -1.15829494e+00, -2.55315347e-02,  3.02805452e-01, -6.00562940e-01, -1.45983889e-01, -1.48857591e+00,  8.02879365e-01, -1.33801206e+00,  1.19863871e+00, -2.46638957e-01, -8.59775344e-01, -9.68340808e-02,  1.11266019e+00,  6.24365245e-01,  3.11990819e-01,  1.47253735e+00, -1.53769193e+00,  1.53856716e+00, -2.76130505e-01,  1.75291122e+00, -4.35464600e-01, -3.47195007e-01, -8.94942605e-01, -7.76755985e-01, -6.07552459e-01, -5.08609502e-01, -1.33836908e+00, -7.83665557e-01,  1.42860343e-01, -1.55696504e+00,  5.44228226e-01, -1.29674711e+00,  1.07030058e+00,  8.50633804e-01, -4.12604508e-01,  9.48976805e-01, -5.44061946e-01,  4.59300206e-01,  4.95492826e-01, -9.56406828e-01,  6.52683281e-01, -8.54735296e-01, -1.87444199e+00,  2.17913290e+00, -2.24228016e+00,  1.62299107e+00,  9.65078714e-01,  7.22093068e-01, -6.60869547e-01, -4.54346543e-01,  3.34855265e-01, -1.99037774e+00,  1.89680510e-01,  1.79781381e-01,  1.05546503e+00, -8.10359027e-01, -1.21485378e-03,  7.32580521e-01, -1.98210896e+00,  7.60775464e-01, -2.94868259e+00,  3.50564473e-01, -3.24647138e-02, -5.05751921e-01, -4.83991684e-01,  4.54351536e-02, -1.53882346e-01,  6.75191839e-01,  7.62140239e-01,  1.96741675e-01,  3.26687445e-01,  1.31490815e+00, -1.06012734e-01, -9.80912944e-01, -1.00670533e+00,  7.79759227e-01, -3.55978454e-01,  4.48341280e-01, -1.11498603e+00, -1.15108937e+00, -4.47250065e-02, -1.68492074e+00, -8.59254840e-01, -6.15441315e-01, -2.67136649e-01,  5.98275502e-01, -1.20312489e-02, -3.28877992e-01, -1.44795475e-01,  4.57283057e-01, -2.24337768e+00,  6.24808834e-01,  3.53901019e-02,  2.09367239e-02, -1.93550863e+00, -1.00982737e+00,  1.05448081e+00]
    x2_coords = [-2.81939258e-02, -2.66112004e+00, -1.30163615e+00, -1.26341759e+00, -4.11430185e-02,  3.12433035e-02,  3.79708431e-01, -3.47864336e-01,  1.00036340e-01,  8.65149051e-01, -1.07896517e+00,  5.54279136e-03,  7.09176051e-01, -5.20454377e-01, -2.20349067e-01, -7.49006222e-01,  5.50899177e-01, -2.71203059e-01,  1.32722214e+00,  2.13592736e+00,  3.31469613e-01, -8.39737702e-03, -1.71054783e+00, -5.71881703e-01, -1.05414827e+00, -1.18879328e+00, -4.95370912e-01, -7.30262649e-01,  2.49412263e-01,  1.07510730e+00, -5.96444978e-01,  1.21850577e-01,  1.60031956e-01, -1.45691424e+00, -1.13079960e+00, -2.88549229e+00, -3.12949076e-01,  7.32662253e-01,  1.57242288e+00,  1.22466586e+00,  3.91857568e-01,  7.21992560e-01, -6.53662162e-01,  3.99689014e-01, -2.02040822e-01, -1.40758509e+00,  1.22124062e+00,  6.12269236e-02,  3.19223803e-01, -6.62044759e-02,  1.12049413e+00, -2.59718462e-01,  5.53603683e-01,  3.43577340e-02, -1.71428055e+00, -8.02648440e-01,  3.41437232e-01, -1.48864330e+00, -6.50486245e-02,  1.74316355e+00, -1.35951326e+00, -4.30914299e-01,  5.55394348e-01, -1.51340384e+00, -8.52209501e-01,  7.45795514e-01,  2.66398888e-01, -7.07713543e-01, -7.48932971e-01,  1.30702074e+00, -5.38239694e-02, -1.77503604e+00,  5.51947608e-01,  1.01435228e+00, -3.84482159e-01,  4.20758352e-01,  2.47100750e-02, -6.73089403e-01,  1.16080300e+00,  3.09846870e-01, -5.03149511e-01,  2.90214969e-01, -1.26009358e+00,  1.07809646e+00,  1.32717380e+00,  7.86415454e-02,  1.06695901e+00,  1.07957932e+00,  2.71056139e+00,  3.59296894e-01,  1.12390481e+00,  1.24594161e+00,  2.59741279e-01, -3.99445901e-02,  1.18982365e+00,  2.56996121e-01,  5.12283538e-02,  7.44703406e-01,  1.06923623e+00,  4.96713778e-01]
    # Plot the given coordinates and values
    y_values = [1.36059385e+01,-2.14295646e+02,-1.40262299e+02,-1.67628889e+02,
     -4.95350234e+00, 1.47504909e+01, 1.45455892e+01,-3.96640850e+01,
     -4.65957315e+01, 1.15054645e+02,-1.56314091e+02, 4.57563622e+01,
      6.01198654e+01,-8.36495061e+01,-2.52026287e+01,-3.15179772e+01,
      7.75096702e+01,-1.45622006e+01, 1.85810417e+02, 1.51200349e+02,
      9.08271135e+01,-1.13116095e+01,-1.01028367e+02,-7.24940737e+01,
     -1.16447227e+02,-1.50316041e+02,-7.78938272e+01,-9.45015864e+01,
      5.27233421e+00, 5.45513944e+01,-8.79361077e+01, 1.72068268e+01,
     -4.32510068e+01,-1.21920517e+02,-1.59807233e+02,-2.42178000e+02,
      1.49769041e+00, 5.62002800e+01, 1.90164685e+02, 9.94256275e+01,
      5.58370049e+01, 8.93974251e+01,-1.00194236e+02, 6.40867007e+01,
     -5.21137701e+01,-2.08786013e+02, 2.02137224e+02,-7.90963707e+01,
      9.28033749e+01, 2.99602590e+01, 1.37030680e+02,-5.04134286e+01,
      3.70521238e+01, 1.61453450e+01,-2.43374292e+02,-7.16253582e+01,
      4.01273714e+01,-1.05743087e+02,-3.68315422e+01, 1.70682163e+02,
     -1.05466359e+02,-1.17341333e+02, 8.30562644e+01,-2.59817753e+02,
     -7.00562560e+01, 7.19039620e+01, 6.91089860e+00,-8.76421302e+01,
     -7.16901421e+01, 1.22233767e+02, 2.04184025e+01,-1.45000695e+02,
      6.14813408e+01, 1.11624902e+02, 1.20554026e+01, 3.70635277e+01,
     -3.47413327e+01,-1.03810923e+02, 1.43276001e+02, 1.70543205e+01,
     -3.21868247e+01,-1.39473401e+01,-1.66999951e+02, 1.03898988e+02,
      6.61720608e+01,-2.48268666e+01, 8.11433527e+01, 9.56334351e+01,
      2.88102138e+02, 3.47516757e+01, 9.74709653e+01, 1.16515731e+02,
      4.26935857e+01,-8.88602062e+01, 1.40263862e+02, 2.63352491e+01,
      5.82156035e+00,-2.61761364e-01, 6.64632902e+01, 8.86441473e+01]
    
    #My Gradient descent implementation estimation.
    #thetas = [-0.14762872238131552, 37.634248895255546, 96.121192425805660]

    thetas = [0.0045479817623415453, 37.863622039676606, 97.943662081800454 ]

    #Tensorflow gradient descent parameter estimation.
    thetas2 = OptimizeGD(np.column_stack((x1_coords, x2_coords)), y_values, 3)

    PlotHypothesis(ax, x1_coords, x2_coords, thetas, y_values, 'red')
    PlotHypothesis(ax, x1_coords, x2_coords, thetas2, y_values)

    plt.show()

def PlotTestGradientDescentEvaluation10Features(fig):
    ax = fig.add_subplot(111, projection='3d')

    # Define the range of the function to be centered around the scattered points
    x_values = 	  [[-1.41638935e+00,  9.13473315e-01, -2.73258644e-01,
                    -5.13639628e-01,  4.87145515e-01, -1.88989647e+00,
                     6.61850812e-01, -2.37005308e+00, -5.91465837e-01,
                     7.04539839e-01],
                   [-7.33338233e-01, -5.82503423e-01,  1.10168490e-01,
                     4.30921497e-01,  3.51033499e-01,  9.28237433e-02,
                    -1.04000380e+00,  5.01720717e-01, -1.91022871e+00,
                     6.05982400e-01],
                   [-1.05886271e-01, -8.05106883e-01,  2.87297937e-01,
                     9.77401978e-01, -1.33443874e-02, -9.78217786e-01,
                    -1.33612812e+00, -2.65199043e-01, -1.32768752e+00,
                     1.47183544e-01],
                   [ 5.44891530e-01, -7.93809542e-01,  4.54135805e-01,
                    -1.86669411e+00, -2.12701437e+00, -1.68278319e-01,
                     6.20502029e-02,  9.57136127e-01, -2.99717904e-01,
                     1.04506974e-01],
                   [-6.57166760e-01, -2.51685509e+00, -4.82754130e-01,
                     4.47815196e-01,  5.49667769e-02, -1.07931146e+00,
                     1.63335797e+00,  1.29506932e-01, -1.86910436e+00,
                     7.94067204e-01],
                   [ 2.41734977e-01, -6.31242363e-01, -6.03971868e-01,
                     1.57537229e-01,  6.50744503e-01, -4.65875653e-01,
                     4.23591366e-01, -4.29691646e-01,  7.61598495e-01,
                     3.35006373e-01],
                   [ 1.17506857e-01, -1.14571290e-01,  2.35976542e-01,
                     6.70390474e-01, -4.87526476e-01, -1.36539252e+00,
                    -5.03827096e-01,  1.30757201e+00,  2.89576576e+00,
                    -5.57167365e-01],
                   [-3.93763434e-01, -3.35451282e-01, -1.72218309e+00,
                     1.68220153e-01, -6.86311506e-01, -4.62617402e-01,
                     7.72927767e-01,  4.15385340e-01,  8.83389325e-01,
                     9.20701804e-01],
                   [ 3.03011398e-01,  3.06737531e-02, -3.43880598e-01,
                    -1.42650981e+00,  1.70400278e+00,  1.66487683e-01,
                     1.28482177e-01,  5.30243377e-01,  5.76279483e-01,
                    -8.15290268e-01],
                   [ 7.05370008e-01,  4.47973329e-01,  1.28574185e+00,
                     2.60054482e+00, -8.91856135e-01, -1.63732214e+00,
                     1.07536511e+00, -3.11031093e-02, -2.09099492e-02,
                     4.29915060e-01],
                   [ 1.98106865e+00,  7.89449355e-01, -2.27276934e-01,
                     7.55157333e-01, -3.58610431e-01,  3.89855753e-01,
                     1.69462554e+00, -7.19484960e-01,  1.91205500e+00,
                    -6.55992747e-01],
                   [-1.63992953e+00,  1.56595742e+00, -6.86203703e-01,
                     4.44619283e-01,  3.84807182e-04, -1.98365982e+00,
                     9.63391545e-01,  2.59021619e-01,  2.27352421e+00,
                     3.11145759e-02],
                   [-2.42925097e-01,  8.94631071e-01,  1.75743647e+00,
                    -7.25241657e-01,  8.01924434e-01,  8.89998149e-01,
                    -1.24101958e+00,  6.67322074e-02, -3.31847698e-01,
                    -1.04883216e+00],
                   [-1.22592934e+00,  1.25626791e+00, -2.25664864e-01,
                    -3.89518659e-03,  1.11986992e+00, -8.42851559e-01,
                     2.90463016e-01, -8.97056774e-01, -1.15345053e+00,
                    -1.03113396e+00],
                   [-1.12573341e+00, -9.18487193e-01, -1.05391030e+00,
                    -2.71793592e-01,  5.10604645e-01, -3.08592325e-01,
                    -3.88127052e-02,  2.86596594e-01,  1.43727299e-01,
                    -3.27970120e-01],
                   [ 5.50391200e-01,  9.97877651e-01,  1.12655241e+00,
                     1.10902893e+00, -3.53407276e-01,  1.20282691e-01,
                     5.85654832e-01, -1.81996943e-01, -2.32342685e+00,
                    -3.77335541e-02],
                   [-1.21721044e-02,  2.09030995e+00,  2.43035876e-01,
                     4.09890949e-01, -4.15852116e-01, -1.37673874e+00,
                     7.11365021e-02, -1.55632049e+00, -2.63159685e-01,
                    -1.61928907e-01],
                   [-6.58807150e-01,  5.36737207e-01, -6.15734242e-01,
                    -7.94447213e-02,  2.92417769e-01,  3.75869996e-01,
                     2.17318600e+00,  4.14093797e-02,  9.34569594e-01,
                     2.19897746e-01],
                   [-4.31987470e-02,  3.48793326e-01,  3.27483400e-01,
                    -7.29277586e-01,  4.95211282e-01, -8.15338613e-02,
                    -1.23147920e+00,  4.22820360e-01,  1.26427852e+00,
                    -3.93806776e-01],
                   [ 1.59236578e+00,  1.00830844e+00, -1.55808949e-01,
                    -1.19375963e+00, -8.58695570e-01,  1.26240085e+00,
                    -1.52781906e+00,  5.66483206e-01,  4.68892086e-01,
                    -6.56925780e-01],
                   [-2.60409483e-01, -1.63163166e+00, -2.49104938e-01,
                    -4.95626845e-01,  6.55558812e-01,  3.75278307e-01,
                    -1.50327135e+00, -4.19810669e-02,  1.52440827e+00,
                     2.76322244e-02],
                   [-4.07684742e-01,  1.02695134e+00, -3.41788550e-01,
                     7.84897548e-02, -9.08667943e-01,  7.56861831e-02,
                    -1.34440808e+00,  2.55617824e-01, -4.37971239e-01,
                    -8.60204435e-01],
                   [ 9.85209378e-01, -6.96259392e-03,  6.02121499e-01,
                     1.13562112e+00, -7.55127610e-01, -4.30169078e-01,
                     1.31433625e+00,  1.38233225e+00, -4.18192532e-01,
                     1.06530703e-01],
                   [-1.10615098e-01, -2.86328446e+00, -3.98368375e-01,
                    -7.59208275e-01, -2.00231007e+00,  1.02166945e+00,
                     6.99484459e-01, -3.27388190e-01, -1.44155011e+00,
                     1.05535209e+00],
                   [-4.75874536e-01, -1.67228385e+00,  3.65351951e-02,
                     1.09532979e+00, -5.44400197e-01,  1.02688799e+00,
                     7.63657496e-01, -5.50267324e-01,  5.33804404e-01,
                    -1.19384888e+00],
                   [-3.15341397e+00,  7.93114467e-01, -2.63407215e+00,
                    -2.36335320e-01,  4.42236697e-01,  2.60370924e-01,
                    -5.57106537e-01,  6.85520935e-01,  1.63999565e+00,
                     2.01031417e+00],
                   [-6.33441315e-01, -1.35350561e+00, -1.52455513e+00,
                    -4.02869285e-01, -4.63903032e-01, -1.54313933e-01,
                    -2.92871509e+00, -7.81214238e-01,  3.45253605e-01,
                    -2.95442495e-01],
                   [ 2.02535016e-01, -1.46585240e+00,  2.35306483e+00,
                    -2.97619564e-02,  4.33875214e-01, -2.94456734e-01,
                     3.15694558e-01, -1.31811069e+00,  1.32451057e+00,
                    -4.58057956e-01],
                   [-1.41431733e+00,  1.03498993e+00, -1.20537975e+00,
                     1.32188667e+00, -1.62693250e+00, -2.07398883e-01,
                    -4.88433381e-01, -9.33870325e-01,  5.88083165e-01,
                     2.78963230e-01],
                   [ 2.61159901e-01, -3.45573319e-03,  8.94441098e-01,
                     1.36612811e+00,  4.94642266e-01, -8.59438630e-01,
                    -6.43864162e-01,  2.71823404e-02,  1.46321151e+00,
                     1.83494234e+00],
                   [-2.63350290e-01, -1.09210424e+00,  9.63530199e-02,
                     2.52368027e+00, -3.72900308e-01,  1.58693724e+00,
                     2.82142885e+00, -2.31077099e-01, -7.98238412e-01,
                     5.94570621e-01],
                   [-1.11626740e+00, -1.39408648e+00, -2.69435833e-01,
                     1.16345766e+00, -1.10808739e+00,  6.17053410e-01,
                    -1.67099307e+00, -3.87780517e-01, -6.27222558e-01,
                    -6.49497871e-01],
                   [-2.49230042e-01, -7.06038589e-01, -3.78697841e-01,
                    -3.42557659e-01,  2.25955265e-01,  9.02162733e-01,
                     8.15969686e-01, -1.11633208e-01, -9.91484726e-01,
                    -6.30487980e-01],
                   [-2.22196204e+00, -1.96236910e+00, -2.86404119e-01,
                    -8.44629717e-02,  7.46598562e-01,  6.28399767e-03,
                     7.15776664e-01, -1.57454280e+00,  3.19363791e-01,
                     6.79228551e-01],
                   [-9.59495777e-01,  7.59590065e-01, -8.61564813e-01,
                    -9.79349651e-01,  2.21067746e-01,  2.25765411e+00,
                     1.10499479e+00, -1.92961263e-01,  3.26669104e-01,
                     8.19212237e-01],
                   [ 4.76470578e-01, -1.25224718e+00, -5.56874936e-01,
                    -1.49604515e-01, -2.23947499e+00,  4.43196347e-01,
                     1.46597655e-01,  2.69182689e-01,  7.00006522e-01,
                    -2.60319197e-01],
                   [ 1.46324333e+00,  2.03329086e-02, -1.63463018e-01,
                    -1.09739021e+00, -1.82553356e+00, -5.85340436e-01,
                    -1.95843186e-01,  7.03703775e-01, -6.07743481e-01,
                    -8.06808858e-01],
                   [-1.12996349e+00, -7.79333480e-02,  2.76083535e+00,
                     1.49894075e+00,  6.79363493e-02,  9.87815415e-01,
                    -8.37176960e-02, -1.42857459e+00,  8.30671908e-01,
                     3.87473467e-01],
                   [-2.08003000e+00, -4.70402176e-01,  9.85703143e-01,
                     2.73568427e+00, -7.76310512e-01, -7.68824438e-01,
                     7.31922086e-02, -2.05993222e+00, -2.85754985e-01,
                    -1.55000537e+00],
                   [ 6.36291895e-02, -6.98308222e-01, -8.75827713e-01,
                     7.55600815e-01,  4.22304051e-01,  3.47840015e-01,
                     4.45660829e-02, -1.41453498e+00,  6.93209288e-01,
                     1.49809333e+00],
                   [-1.13878153e+00, -6.66301472e-01,  8.84830935e-01,
                    -1.24737233e+00, -8.02001562e-01, -4.79634400e-01,
                    -6.71218690e-01,  7.54642724e-01,  1.76775833e-01,
                     1.34110895e+00],
                   [-1.79192935e+00, -5.51819542e-01,  1.58701319e+00,
                     1.27512530e+00,  2.68504245e-01, -5.82354883e-01,
                     1.55001401e+00, -5.18248325e-01,  7.82208873e-01,
                    -2.03034901e+00],
                   [-7.43925846e-01,  1.32781082e+00,  3.77934031e-01,
                     8.48386175e-01,  9.74930535e-01, -2.77012016e-01,
                    -1.34163450e+00, -7.46800128e-01,  1.54581226e+00,
                    -2.25725559e-01],
                   [-5.07880681e-01,  5.13680598e-01, -2.07692492e+00,
                    -7.69485178e-01,  1.41027185e+00,  1.18304015e+00,
                    -5.14777488e-02, -1.73286058e-01,  1.76043146e-01,
                    -4.34727495e-01],
                   [ 4.84230918e-02, -6.03958832e-01,  5.80363500e-01,
                     1.44920264e-01, -6.88254965e-01,  1.35535876e+00,
                    -1.89768780e-01, -4.20385420e-01, -5.51111835e-01,
                    -1.09077444e+00],
                   [-1.30073745e+00, -3.82662517e-01,  1.99666401e-01,
                    -4.58831181e-01,  6.33783252e-01,  1.06245787e+00,
                    -1.07868005e+00, -7.31761025e-01, -1.66915788e-01,
                     8.16746409e-01],
                   [-1.42138709e+00, -8.95550602e-01,  1.00467165e+00,
                    -7.84920237e-01,  1.66116429e-01, -6.53546285e-01,
                     1.68245496e+00, -3.27333884e-01,  1.11286363e+00,
                     1.86792789e-01],
                   [ 5.89530658e-01,  7.63758170e-01, -6.68230229e-01,
                     7.94844022e-01,  9.72575016e-01, -1.82472013e+00,
                    -2.40392013e+00, -3.31603472e-01, -2.73132916e+00,
                     1.92618894e+00],
                   [-5.82447112e-01,  2.32297871e+00,  2.66088681e-01,
                     3.30865133e-01,  1.48074675e+00, -2.76429039e-02,
                     1.75536506e-01, -6.24408113e-01,  1.23622360e+00,
                     2.08472633e-01],
                   [-8.85262957e-01,  3.12450472e-01,  2.97047660e-01,
                     1.50790044e+00,  5.78822981e-01,  4.96668555e-01,
                     1.36600010e+00,  2.12359883e-01,  9.24617628e-01,
                    -6.58296988e-01],
                   [-7.35164532e-01, -2.82605434e-02,  1.31754253e+00,
                    -4.64561084e-01, -3.15083410e-01, -4.09342069e-01,
                    -7.19480760e-01,  4.27639541e-01, -1.15693048e+00,
                    -9.42408403e-01],
                   [ 3.77556385e-01, -5.08800635e-01, -1.48837541e+00,
                     2.21173379e-01, -1.26653797e+00, -1.57110116e-01,
                     1.20320807e+00,  9.99451995e-01, -1.62611448e-01,
                    -4.28163960e-01],
                   [-3.42772914e-01,  1.41730321e+00, -8.89092663e-01,
                    -3.80873074e-02, -2.26197445e-01,  7.17072741e-01,
                     2.32433756e+00,  1.06404954e+00,  5.89373208e-01,
                     1.03393661e+00],
                   [ 3.88234049e-01, -1.36850892e-01,  2.11912569e-01,
                    -1.31916180e+00,  5.78807272e-01, -9.48918262e-02,
                    -1.62839433e+00,  4.30284934e-01, -6.50660240e-01,
                    -3.00822595e-01],
                   [-1.09919767e-01, -1.33218175e+00, -4.87694649e-01,
                    -3.50645700e-01, -4.35156045e-02, -7.31196908e-01,
                     1.69425730e+00,  2.18522868e+00, -5.01633559e-01,
                    -6.42691889e-01],
                   [ 2.85144570e-01,  3.14621517e-01, -2.43036536e-01,
                    -6.36801372e-01,  1.86553242e+00, -1.16979983e-01,
                    -2.04373159e-01,  3.35468908e-01, -1.01561649e+00,
                     3.05491736e-01],
                   [ 8.18437441e-01, -6.78272231e-01,  1.63660199e-01,
                    -1.80875563e+00, -1.34666702e-01, -1.76368039e-01,
                     1.29138367e+00, -3.76424126e-01,  1.39424538e+00,
                     1.17203089e+00],
                   [-6.51714602e-02,  1.68510402e-01, -9.39970942e-01,
                    -1.71398750e+00,  2.31425641e+00,  5.18845982e-01,
                    -3.78273152e-01, -3.09536780e-01,  7.94869054e-01,
                     1.04281276e+00],
                   [ 8.45438135e-01, -9.59005557e-01,  1.36692795e+00,
                     7.90738464e-01, -5.44379569e-02,  1.05748192e-02,
                     1.27214224e+00, -2.20015997e+00,  9.55655715e-01,
                    -2.25741886e+00],
                   [ 2.99019043e-01, -2.92424095e+00,  1.46480870e+00,
                    -3.02847556e-01,  8.51666368e-01,  1.08056690e+00,
                    -3.94380250e-01,  8.90460967e-01, -6.59394305e-01,
                    -5.43603727e-01],
                   [ 2.22551152e+00, -1.51569904e+00,  5.86341151e-01,
                     7.70062014e-02, -1.28577916e-01,  2.29183427e+00,
                     2.70177456e+00, -1.77341390e+00, -7.52392847e-01,
                     5.56246977e-01],
                   [ 7.13755467e-01, -3.30422195e-01,  1.84771356e+00,
                     1.20555732e+00, -6.70219300e-01,  9.93043757e-01,
                     1.20083548e+00,  1.48789042e+00, -8.46867186e-01,
                    -1.90696147e+00],
                   [-1.77762045e-01, -1.76243939e+00, -2.91665678e-02,
                     1.30730502e+00, -1.01712649e+00,  1.01325293e+00,
                     1.42713490e+00, -1.29531295e-01,  6.15567861e-01,
                     1.36010822e-01],
                   [-8.68796890e-01,  1.40868681e+00, -2.30477951e+00,
                     9.80202475e-01, -9.89079268e-01,  7.10435660e-02,
                     3.63514082e-01,  2.64409510e-01, -5.42557140e-01,
                    -4.67663963e-01],
                   [-1.01376104e+00,  1.42915126e-01,  1.28978102e+00,
                     3.16414959e-01, -1.25876252e+00, -7.21343324e-01,
                    -2.62026271e-01,  3.53370554e-01,  1.15074334e-01,
                    -1.25649835e+00],
                   [-7.96666535e-01,  5.24098597e-01, -4.01659368e+00,
                    -1.36311423e+00, -6.03245588e-01, -1.14925453e+00,
                    -2.44000095e-02,  5.36315121e-01, -2.16410703e+00,
                     1.18184263e+00],
                   [-1.95594312e+00,  1.49923916e+00, -5.54452581e-01,
                     5.31056669e-01, -2.11439184e+00,  5.34649367e-01,
                     1.01749141e+00, -4.15749358e-01,  5.25173957e-01,
                     1.43351924e+00],
                   [ 3.85943171e-01,  3.10943237e-01,  6.22162107e-01,
                    -6.01522146e-01, -1.79155898e+00, -1.91055896e+00,
                     1.22697881e+00,  4.19456855e-01,  5.48171318e-01,
                     1.90836517e-01],
                   [ 1.68829675e+00, -7.89203967e-01,  3.28113138e-01,
                    -4.01554487e-01, -4.95192349e-01, -7.18627358e-03,
                    -7.75463473e-01, -3.88409637e-01,  9.04542607e-01,
                     1.06443804e+00],
                   [-9.20754614e-01,  4.15609766e-01, -5.10460340e-01,
                     3.38491809e-01,  7.00945807e-01,  1.95298100e+00,
                     5.37299746e-01, -2.24847196e+00,  4.84033384e-01,
                     5.08047213e-01],
                   [ 5.13661035e-01,  9.66696539e-01,  9.08801240e-01,
                     1.53021468e+00, -2.65131970e-02, -5.20341220e-01,
                    -6.87871429e-01,  1.75143691e+00, -6.43279069e-01,
                    -1.03053323e-01],
                   [ 4.82129577e-01,  3.98611646e-01,  2.96774925e-01,
                     7.92646410e-01,  2.64591437e-01,  5.74378372e-01,
                     1.07555140e+00,  1.62602193e+00, -8.71572119e-02,
                     7.29751131e-01],
                   [ 1.64981570e-01, -1.11996777e+00, -8.95348229e-01,
                    -1.04901657e+00,  2.19879912e-01,  1.97004697e-01,
                     2.57010353e-01, -1.50684316e+00, -9.07243945e-01,
                     1.25586959e-01],
                   [-9.26117889e-01,  2.45171973e-01,  7.10586977e-01,
                    -8.15168390e-01,  1.53570760e+00, -2.86360710e-01,
                     5.80940143e-01,  9.98273883e-01, -1.70475723e+00,
                    -4.13884573e-01],
                   [ 5.39729046e-01,  1.11094695e+00,  8.59264008e-01,
                    -1.69926468e+00, -5.73189472e-01, -2.98457682e-01,
                     2.96446922e-01,  1.12829766e+00, -1.69635426e+00,
                    -1.45786429e+00],
                   [-6.44324147e-01,  2.50210416e+00,  5.54871762e-01,
                     1.43396111e+00, -1.09235041e+00,  9.90767778e-01,
                     2.10820557e-01,  2.95772878e-01,  5.46986765e-01,
                    -8.31576840e-01],
                   [ 4.81165237e-01,  3.18460807e-02,  7.05320798e-01,
                    -1.14065872e+00, -8.55008482e-01, -2.46662858e-01,
                    -8.13425246e-01, -1.52504708e+00,  4.41189171e-01,
                    -1.56774562e+00],
                   [-1.38421814e-01, -8.99584114e-01,  1.21060403e+00,
                     1.52508376e+00,  3.02352791e-01, -1.23710085e+00,
                     3.67204916e-01, -1.60159560e+00,  9.23902757e-01,
                    -1.60243875e+00],
                   [ 1.26075977e+00, -1.69657725e+00, -2.86683695e+00,
                     2.74234632e-02,  1.03022750e+00, -1.77741981e-01,
                    -1.06626159e-01,  1.77311206e+00,  1.83236179e-01,
                     3.52234957e-01],
                   [-4.07171580e-01,  5.96648471e-01,  1.23625776e+00,
                    -2.23801445e-01,  3.65357362e-02, -2.50203756e-01,
                    -1.22012278e+00,  6.29976664e-01,  4.22683036e-02,
                    -3.92068568e-01],
                   [-7.64648732e-01,  4.24208725e-01,  2.81019675e-02,
                     1.18108937e+00, -6.34041561e-01, -1.93496596e-01,
                    -3.76721695e-02, -9.23068377e-04, -2.26332958e-01,
                    -5.03953068e-01],
                   [-3.25350269e-01,  2.57645156e-01,  6.63412912e-01,
                     1.65028305e+00, -3.58173517e-01,  1.26595335e+00,
                     5.58381683e-01, -2.55586625e+00,  1.51884729e+00,
                    -6.11025320e-01],
                   [-7.25253906e-01, -1.09120917e+00,  4.99163888e-01,
                     3.31139137e-01,  2.38149560e-01,  7.90410923e-01,
                     3.27382744e-01,  1.38686218e-01,  1.57979599e+00,
                     1.60499990e+00],
                   [-1.06381013e-01, -4.82638710e-01,  7.28941852e-01,
                    -6.19196643e-01,  9.93410415e-01, -6.23346254e-02,
                    -1.11864350e-01, -7.46148204e-01, -6.27226830e-01,
                     7.42168055e-01],
                   [-7.22707179e-01,  2.97283865e-01,  2.87498300e-01,
                    -9.04494657e-01, -1.06141945e-01,  8.44512723e-01,
                    -3.75275966e-01, -1.86125208e-01, -1.56819635e-01,
                     7.01384415e-01],
                   [ 7.95653749e-01, -1.31458717e+00, -6.80709079e-01,
                     7.24922231e-01,  3.88420708e-01, -5.50558771e-01,
                    -4.85452932e-01, -1.31924524e+00, -7.82261255e-01,
                     1.29934585e+00],
                   [-1.54065083e-01,  1.38315648e+00,  6.53867999e-01,
                     1.38041422e+00,  4.72653857e-01, -8.76890226e-01,
                    -1.73061247e+00,  6.30560572e-01,  8.76407091e-02,
                    -2.88265392e+00],
                   [-2.31555882e+00,  9.34145369e-01,  1.58916408e-01,
                     1.95429456e-03, -1.00189421e+00, -2.82560545e+00,
                    -8.20229953e-01, -1.52255921e+00, -1.63569787e+00,
                     9.47639883e-01],
                   [ 8.36223760e-02,  1.06007993e+00, -8.60366372e-01,
                     3.18632890e-02, -6.31399673e-01, -8.95292921e-02,
                     1.75892103e+00,  5.36496859e-01,  2.34148977e+00,
                     3.10400215e-01],
                   [-3.90827008e-01, -4.04582817e-01, -4.65207275e-01,
                     3.91702802e-01, -1.19959491e+00,  1.31478886e+00,
                    -5.13417489e-01,  2.09397166e-01, -1.64263920e+00,
                    -1.05646010e+00],
                   [-6.77716931e-01,  9.87856333e-01, -1.61558420e+00,
                    -1.42108002e-01, -3.20673177e-01,  1.66380533e-01,
                     1.60558808e+00, -2.90085478e-01, -1.15275417e+00,
                    -1.59440290e+00],
                   [ 6.23752314e-01, -5.44173934e-01, -6.24828108e-01,
                     8.24519106e-01,  5.53566490e-01, -7.02853343e-01,
                     1.40688476e+00, -9.18226968e-01, -1.08114956e+00,
                     8.33121420e-01],
                   [ 1.03963413e-01, -2.85173041e-01,  1.26723662e-01,
                    -3.60904418e-02,  1.03469454e+00,  1.04202370e+00,
                     2.78915124e-01,  1.10764793e+00, -9.77720472e-01,
                     3.81849199e-01],
                   [-1.41115066e+00, -2.08549234e+00, -3.79139539e-02,
                     1.63328279e+00, -3.94673898e-01,  2.37487487e+00,
                     1.59765095e+00,  1.13148981e+00, -4.76669230e-01,
                    -3.07944154e-01],
                   [ 2.38188157e-01,  2.10731780e-01, -8.71008802e-01,
                    -1.23345335e+00, -1.20443465e+00, -7.17572789e-01,
                    -5.41525725e-01, -6.98039415e-01,  1.18101339e+00,
                    -2.20491522e+00],
                   [ 8.79147962e-01,  6.96134002e-01,  5.63830327e-01,
                    -1.57536122e+00, -5.95600476e-01, -1.12228751e+00,
                    -6.28870904e-02,  1.20123282e+00, -1.16027088e+00,
                    -1.81516341e+00],
                   [-6.93775464e-01,  1.30456342e+00, -4.33230370e-01,
                    -7.02487026e-01, -1.63211788e-01, -1.01258746e+00,
                     2.65890436e+00,  4.40220752e-01, -1.40296693e+00,
                     1.02974532e-01],
                   [-4.92775142e-01, -1.28576770e-01, -3.84513875e-01,
                     1.91348893e-01, -1.28733570e-01, -1.52557191e+00,
                    -1.16744606e+00, -1.00052830e+00, -2.95792296e+00,
                    -7.49224357e-02],
                   [-1.88313460e-01,  3.85828031e-01,  8.17201121e-01,
                    -1.11928003e+00,  7.14894153e-01, -9.64538361e-01,
                     5.71484855e-02,  6.20873468e-01, -1.15431216e+00,
                    -8.14563760e-02],
                   [-1.12329969e-01, -5.59128304e-02,  5.14319639e-01,
                    -6.88486342e-01, -2.14406057e-01, -6.61258662e-01,
                    -5.75429597e-01, -1.78582818e+00, -1.98854994e-01,
                    -1.38835537e-01]]
    # Plot the given coordinates and values
    y_values = [-1.81177945e+02, -1.54805752e+02, -2.02874436e+02, -2.57138411e+02,
                -4.53397532e+02, -4.41879073e+01,  1.69790311e+02, -9.81978631e+01,
                -6.03131051e+01,  2.45636881e+02,  3.17743758e+02,  1.63954333e+02,
                1.89511881e+02, -1.08980271e+02, -2.32137778e+02,  9.85832227e+01,
                7.86829518e+01,  1.60780648e+02,  8.74628461e+01,  8.23450736e+01,
                -6.91735873e+01, -4.27733961e+01,  1.07149533e+02, -4.10618086e+02,
                -2.37232431e+00,  6.65918945e+01, -3.85920385e+02,  1.38675202e+02,
                7.07664553e+01,  3.31763495e+02,  2.39121287e+02, -1.73597162e+02,
                -1.51275765e+02, -1.72791400e+02,  2.02400461e+02, -1.44278279e+02,
                -2.72949973e+02,  5.18093063e+02,  3.54946807e+01,  6.46071873e+01,
                -7.86570474e+01,  1.32092724e+02,  3.18971183e+02, -7.32927708e+01,
                7.82453425e-03,  3.77017094e+01,  1.13634809e+01, -2.58031375e+02,
                4.25173150e+02,  3.10951629e+02, -1.21695435e+02, -2.10527497e+02,
                2.70257591e+02, -1.86831237e+02, -2.84014734e+02, -7.88443461e+01,
                -1.25666253e+01,  4.08112289e+00,  7.24633781e+01, -1.38461675e+02,
                1.12072019e+02,  1.89717861e+02,  7.61057905e+01, -7.83799500e+01,
                7.24736990e+00, -6.79046324e+02,  2.47627919e+02, -8.50615593e+01,
                2.85471856e+01,  2.51483656e+02,  2.07754273e+02,  2.60084836e+02,
                -3.43485413e+02, -1.14052582e+02, -1.79702465e+02,  4.87669562e+02,
                -1.33007194e+02,  3.23655774e+01, -3.56441943e+02,  1.02310242e+02,
                5.61839587e+01,  4.00740584e+02,  2.57902818e+02, -4.49057395e+01,
                6.03980587e+01, -2.05178730e+02,  8.97597340e+01, -3.22176568e+02,
                2.67343899e+02, -1.68639404e+02, -2.09886449e+02, -1.33043612e+02,
                5.15527040e+01,  1.00541303e+02, -2.68157487e+02, -2.84848176e+02,
                -1.28483057e+02, -4.76231974e+02, -1.41211452e+02, -1.37460089e+02]
    
    #My Gradient descent implementation estimation.
    theta = [   0.0058944460135096745, 7.0603130476287417, 99.186929256239296, 92.177165549087306, 
                79.739154048269683, 24.895501507001693, 88.917411592939814, 18.888654531146802,
                13.812211823000036, 88.410687066030206, 46.583938733373316]

    costs = np.dot(x_values, theta[1:]) + theta[0]
    costDiff = costs-y_values

    #Tensorflow gradient descent parameter estimation.
    startTime = time.time()
    thetas2 = OptimizeGD(x_values, y_values, 11)
    endTime = time.time()

    executionTime = endTime - startTime

    print(f"Execution time: {executionTime} seconds")

    print(f"Thetas2: {thetas2[1:]}")
    costs2 = np.dot(x_values, thetas2[1:]) + thetas2[0]

    costDiff2 = costs2-y_values

    print(costs2)

#print("******* Executing the script ********")
def MethodToCall():
    print("************ MethodToCall *************")
    fig1 = plt.figure()
    PlotTestGradientDescentEvaluation2Features(fig1)
    print("************ MethodToCall Exiting *************")
    plt.show()


#fig1 = plt.figure()
#PlotTestGradientDescentEvaluation2Features(fig1)
##PlotTestGradientDescentEvaluation2Features2(fig1)
##PlotTestGradientDescentEvaluation10Features(fig1)

#plt.show()

#print("******* Execution completed ********")




