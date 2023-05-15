import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
m = np.array([1,1,1,1,1,1,1,1,1,1,1])
# x = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
# y_our = np.array([10,10,10,10,10,10,9,9,9,8,6])/10
# y_dso = np.array([10,10,10,9,9,8,7,7,6,5,3])/10
# y_gp = np.array([10,10,9,5,5,4,2,2,1,0,0])/10
# y_scale = np.array([10,2,1,0,0,0,0,0,0,0,0])/10
# y_EQL = np.array([10,10,8,7,4,2,2,1,0,0,0])/10
#
# plt.figure(figsize=(12.8, 7.2))
# plt.plot(x,y_our,c = '#32CD32',marker="o",linestyle ='-',label = 'our',linewidth=5,markersize = 16)
# plt.plot(x,np.mean(y_our)*m,c = '#32CD32',linestyle ='--',label = 'our_mean',linewidth=2,markersize = 16)
# plt.plot(x,y_dso,c = '#87CEFA',marker="d",linestyle ='-',label = 'DSO',linewidth=5,markersize = 16)
# plt.plot(x,np.mean(y_dso)*m,c = '#87CEFA',linestyle ='--',label = 'DSO_mean',linewidth=2,markersize = 16)
# plt.plot(x,y_gp,c = '#4169E1',marker="s",linestyle ='-',label = 'GP',linewidth=5,markersize = 16)
# plt.plot(x,np.mean(y_gp)*m,c = '#4169E1',linestyle ='--',label = 'GP_mean',linewidth=2,markersize = 16)
# plt.plot(x,y_scale,c = '#483D8B',marker="^",linestyle ='-',label = 'NeSymReS',linewidth=5,markersize = 16)
# plt.plot(x,np.mean(y_scale)*m,c = '#483D8B',linestyle ='--',label = 'NeSymReS_mean',linewidth=2,markersize = 16)
# plt.plot(x,y_EQL,c = '#FFD700',marker="+",linestyle ='-',label = 'EQL',linewidth=5,markersize = 16)
# plt.plot(x,np.mean(y_EQL)*m,c = '#FFD700',linestyle ='--',label = 'EQL_mean',linewidth=2,markersize = 16)
# plt.xlabel('Noise level',size = 20)
# plt.ylabel('Recovery rate(n/100)',size = 20)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.)
# plt.savefig("noise.pdf")
# plt.show()

x = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
y_our = np.array([114,107,102,92,94,90,87,88,76,72,69])/120
y_dso = np.array([111,106,104,96,90,82,80,74,68,49,44])/120
y_dsr = np.array([100,86,79,66,57,53,50,44,42,45,38])/120
y_gp =  np.array([73,72,68,64,52,54,46,44,36,37,32])/120
y_scale = np.array([78,44,24,15,13,14,16,12,11,14,11])/120


plt.figure(figsize=(12.8, 8))
# 绘制折线图
# sns.set(style="ticks")
# sns.set_palette("husl")
# fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(x,y_our,c = '#32CD32',marker="o",linestyle ='-',label = 'our',linewidth=5,markersize = 16)
plt.plot(x,np.mean(y_our)*m,c = '#32CD32',linestyle ='--',label = 'our_mean',linewidth=2,markersize = 16)
plt.plot(x,y_dso,c = '#87CEFA',marker="d",linestyle ='-',label = 'DSO',linewidth=5,markersize = 16)
plt.plot(x,np.mean(y_dso)*m,c = '#87CEFA',linestyle ='--',label = 'DSO_mean',linewidth=2,markersize = 16)
plt.plot(x,y_gp,c = '#4169E1',marker="s",linestyle ='-',label = 'GP',linewidth=5,markersize = 16)
plt.plot(x,np.mean(y_gp)*m,c = '#4169E1',linestyle ='--',label = 'GP_mean',linewidth=2,markersize = 16)
plt.plot(x,y_scale,c = '#483D8B',marker="^",linestyle ='-',label = 'NeSymReS',linewidth=5,markersize = 16)
plt.plot(x,np.mean(y_scale)*m,c = '#483D8B',linestyle ='--',label = 'NeSymReS_mean',linewidth=2,markersize = 16)
plt.plot(x,y_dsr,c = '#FFD700',marker="+",linestyle ='-',label = 'DSR',linewidth=5,markersize = 16)
plt.plot(x,np.mean(y_dsr)*m,c = '#FFD700',linestyle ='--',label = 'DSR_mean',linewidth=2,markersize = 16)

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.tick_params( length=10, width=4, direction='out')
plt.xlabel('Noise level',size = 23)
plt.ylabel('Recovery rate (n/120)',size = 23)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.,fontsize=11)
plt.legend(ncol=4,loc='upper right',fontsize=12,frameon=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['bottom'].set_linewidth(4)
plt.tight_layout()
plt.savefig("noise2.pdf")
plt.show()




# x = np.array([2,4,6,8,10,12,14,16,18,20])
# y_our_down = np.array([0.99,0.991,0.98,0.982,0.983,0.986,0.986,0.982,0.97,0.95])
# y_our =      np.array([0.999,0.997,0.992,0.998,0.991,0.995,0.991,0.988,0.982,0.973])
# y_our_up =   np.array([0.9999,0.999,0.998,1.0,0.998,1.0,0.996,0.999,0.992,0.99])
# y_scale =      np.array([0.66,0.79,0.82,0.78,0.76,0.73,0.67,0.43,0.23,0.22])
#
# plt.plot(x,y_our,color="#4682B4",linewidth=2,marker="o",markersize=9,
#          markerfacecolor="#4682B4",markeredgewidth=1,markeredgecolor="w",label = 'our')
# plt.fill_between(x, y_our_down, y_our_up, facecolor='#4682B4', alpha=0.3)
# plt.plot(x,y_scale,'ro-.',label = 'NeSymReS',markersize = 6)



# x = np.array([2,4,6,8,10,12,14,16,18,20])
# y_our =      np.array([0.999,0.997,0.992,0.998,0.991,0.995,0.991,0.988,0.982,0.973])
#
# y_scale_down = np.array([0.54,0.66,0.77,0.72,0.64,0.62,0.56,0.22,0.05,0.02])
# y_scale =      np.array([0.66,0.79,0.82,0.78,0.76,0.73,0.67,0.43,0.23,0.22])
# y_scale_up =   np.array([0.78,0.85,0.88,0.86,0.82,0.79,0.76,0.61,0.56,0.66])
#
# plt.plot(x,y_our,'ro-.',label = 'our',markersize = 6)
# plt.plot(x,y_scale,color="#4682B4",linewidth=2,marker="o",markersize=9,
#          markerfacecolor="#4682B4",markeredgewidth=1,markeredgecolor="w",label = 'NeSymReS')
# plt.fill_between(x, y_scale_down, y_scale_up, facecolor='#4682B4', alpha=0.3)

# plt.xlim(0,22)
# plt.xlabel('Ranges',size = 14)
# plt.ylabel('Goodness of Fit (R2)',size = 14)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=3, mode="expand", borderaxespad=0.)
# # plt.savefig("range1.pdf")
# plt.savefig("no2.pdf")
# plt.show()



# ####柱状图
#
# # ==========================子图（b）====================================
# index_fb1=[0.1,0.5,0.9,1.3]
# index_fb2=[0.2,0.6,1.0,1.4]
# # index_fb3=[0.3,0.7,1.1,1.5]
#
# fb1 = [42,492,108,248]
# fb2 = [96,823,212,420]
# # fb1 = []
# plt.bar(index_fb1, fb1, width=0.1, label=  'The bar With '+'\u03B1'+'-ada', color='#87CEFA', zorder=1)
# plt.bar(index_fb2, fb2, width=0.1, label='The bar No '+'\u03B1'+'-ada', color='#4682B4', zorder=1)
# plt.plot(index_fb1,fb1,c = 'g',marker="o",linestyle ='--',label = 'The val With '+'\u03B1'+'-ada',linewidth=3,markersize = 10)
# plt.plot(index_fb2,fb2,c = 'y',marker="d",linestyle ='--',label = 'The val No '+'\u03B1'+'-ada',linewidth=3,markersize = 10)
# plt.legend(frameon=False)
#
# # plt.ylim(0, 1.3)
# plt.xticks([0.15, 0.55, 0.95,1.35], ['Nguyen-2', 'Nguyen-6', 'Nguyen-9','Nguyen-10'], )
#
# plt.xlabel('Expression', fontsize=15 )
# plt.ylabel('Spend time (s)', fontsize=15)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
# ax.spines['top'].set_linewidth(2)
# plt.savefig("bar.pdf")
# plt.show()


# m = np.array([1,1,1,1,1,1,1,1,1,1])
# x = np.array([2,4,6,8,10,12,14,16,18,20])
# y_our_down = np.array([0.99,0.984,0.982,0.982,0.983,0.984,0.976,0.972,0.95,0.94])
# y_our =      np.array([0.999,0.997,0.992,0.998,0.991,0.995,0.991,0.988,0.982,0.973])
# y_our_up =   np.array([0.9999,0.999,0.998,1.0,0.998,1.0,0.996,0.999,0.992,0.99])
# y_scale =      np.array([0.66,0.79,0.82,0.78,0.76,0.73,0.67,0.43,0.23,0.22])
#
# plt.figure(figsize=(12.8, 7.2))
# plt.plot(x,y_our,color="y",linewidth=4,marker="o",markersize=16,
#          markerfacecolor="y",markeredgewidth=1,markeredgecolor="w",label = 'our')
# plt.fill_between(x, y_our_down, y_our_up, facecolor='#4682B4', alpha=0.3)
# # plt.plot(x,np.mean(y_our)*m,c = 'y',linestyle ='--',label = 'our_mean',linewidth=2,markersize = 10)
#
# y_scale_down = np.array([0.54,0.66,0.77,0.72,0.64,0.62,0.56,0.22,0.05,0.02])
# y_scale =      np.array([0.66,0.79,0.82,0.78,0.76,0.73,0.67,0.43,0.23,0.22])
# y_scale_up =   np.array([0.78,0.85,0.88,0.86,0.82,0.79,0.76,0.61,0.56,0.66])
#
# # plt.plot(x,y_our,'ro-.',label = 'our',markersize = 6)
# plt.plot(x,y_scale,color="#4682B4",linewidth=4,marker="o",markersize=16,
#          markerfacecolor="#4682B4",markeredgewidth=1,markeredgecolor="w",label = 'NeSymReS')
# plt.fill_between(x, y_scale_down, y_scale_up, facecolor='#4682B4', alpha=0.3)
# plt.plot(x,np.mean(y_scale)*m,c = 'r',linestyle ='--',label = 'NeSymReS_mean',linewidth=2,markersize = 10)
# plt.plot(x,np.mean(y_scale_down)*m,c = 'b',linestyle ='--',label = 'NeSymReS_up_mean',linewidth=2,markersize = 10)
# plt.plot(x,np.mean(y_scale_up)*m,c = 'g',linestyle ='--',label = 'NeSymReS_down_mean',linewidth=2,markersize = 10)
# plt.xlim(0,22)
# plt.xlabel('Ranges',size = 20)
# plt.ylabel('Goodness of Fit (R2)',size = 20)
# plt.legend(prop={'size': 16})
#
# # plt.savefig("range1.pdf")
# plt.savefig("no2.pdf")
# plt.show()