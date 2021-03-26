import numpy as np
import functools as fts

import PyQt5.QtWidgets as QW
import PyQt5.QtGui as QG
import PyQt5.QtCore as QC
from PyQt5.QtCore import Qt,pyqtSignal,pyqtSlot

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patches
import matplotlib.ticker as ticker

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D,proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import interp2d
from colorsys import hls_to_rgb

from typing import Iterable
from numbers import Number

import gui_lib as gl


def par_comp(u,n):
	return np.einsum("...x,x,y->...y",u,n,n)/np.dot(n,n)

def ort_comp(u,n):
	return u-par_comp(u,n)

def plot(x,y):
	plt.plot(x,y)
	plt.show()

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    c=c/np.max(c)
    return c

def orthogonal_proj(zfront, zback):
	a = (zfront+zback)/(zfront-zback)
	b = -2*(zfront*zback)/(zfront-zback)
	return np.array([[1,0,0,0],
						[0,1,0,0],
						[0,0,a,b],
						[0,0,-0.0001,zback]])
						
def perspective_proj(zfront, zback):
	a = (zfront+zback)/(zfront-zback)
	b = -2*(zfront*zback)/(zfront-zback)
	return np.array([[1,0,0,0],
						[0,1,0,0],
						[0,0,a,b],
						[0,0,-1,0]])
						
def set_orthogonal_projection(flag):
	proj3d.persp_transformation = orthogonal_proj if flag else perspective_proj

def rgba_to_gray(x):
	return np.dot(x[:,:3],[0.299, 0.587, 0.114])
					  
def mappable(vmin,vmax,cmap):
	norm  = mpl.colors.Normalize(vmin,vmax)
	sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	return norm,sm
	
def addCMActions(menu,setter,parent=None):
	for name in [d[:-2] for d in dir(plt.cm) if d.endswith("_r")]:
		action=QW.QAction(name,parent,triggered=fts.partial(setter,getattr(plt.cm,name)))
		menu.addAction(action)

def set_scale(ax,scale_x=1,scale_y=1):
	ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_x))
	ax.xaxis.set_major_formatter(ticks_x)

	ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*scale_y))
	ax.yaxis.set_major_formatter(ticks_y)

def addXYActions(menus,func,event,parent):
	for m in menus:
		for ax in parent.axes:
			if ax.bbox.containsx(event.x) and ax.bbox.containsy(event.y):
				if ax.get_images():
					break
		else:
			continue
		rel="relative" in m.title().lower()
		act_x=QW.QAction("along x in current view",parent,triggered=fts.partial(func,ax,event.xdata,0,rel,m.title()))
		m.addAction(act_x)
		act_y=QW.QAction("along y in current view",parent,triggered=fts.partial(func,ax,event.ydata,1,rel,m.title()))
		m.addAction(act_y)
		
def reverse_cmap(cmap):
	if cmap.endswith("_r"):
		return cmap[:-2]
	return cmap+"_r"

def CustomCmap(from_rgb,to_rgb,name="custom_map"):

    r1,g1,b1 = from_rgb

    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap(name, cdict)
    return cmap

class VikSlider(QW.QSlider):
	play_css,pause_css=("QSlider::groove:horizontal {{ "
					  "border: 1px solid #999999; "
					  "height: 20px; "
					  "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4); "
					  "margin: 2px 0; "
					  "}} "
					  "QSlider::handle:horizontal {{ "
					  "image: url('C:/Users/vjons/Dropbox/PythonLibrary/resources/{}'); "
					  "background: transparent; "
					  "width: 30px; "
					  "margin: -2px 0px; "
					  "}} ".format(img) for img in ("play.png","pause.png"))
	def __init__(self,orientation=Qt.Horizontal,parent=None):
		QW.QSlider.__init__(self,orientation,parent)
		self.playing=False
		self.rate=1
		
		self.setStyleSheet(self.play_css)
		self.sliderPressed.connect(self.handlePressed)
		self.sliderReleased.connect(self.handleReleased)
		
		self.timer = QC.QTimer(self)
		self.timer.timeout.connect(self.shiftHandle)
		
	def shiftHandle(self):
		value=self.value()+int(np.sign(self.rate))
		if self.rate>0 and value>self.maximum():
			value=self.minimum()
		if self.rate<0 and value<self.minimum():
			value=self.maximum()
		self.setValue(value)
	
	def playPause(self):
		self.handlePressed()
		self.handleReleased()

	def handlePressed(self):
		self.downValue=self.value()
		if self.playing:
			self.timer.stop()

	def handleReleased(self):
		if self.downValue==self.value():
			self.playing=not self.playing
			if self.playing:
				self.timer.start(int(1000/np.abs(self.rate)+0.5))
			self.setStyleSheet(self.pause_css if self.playing else self.play_css)
		elif self.playing:
			self.timer.start(int(1000/np.abs(self.rate)+0.5))
			self.setStyleSheet(self.pause_css)
		
	def setRate(self,rate):
		try:
			self.rate=float(rate)
		except:
			return
		if self.playing:
			self.timer.stop()
			if not self.rate:
				self.rate=1e-3
			self.timer.start(int(1000/np.abs(self.rate)+0.5))
			
class VikToolbar(NavigationToolbar):
	toolitems=[t for t in NavigationToolbar.toolitems if t[-1]!="home"]
	
	def __init__(self,canvas,parent=None):
		NavigationToolbar.__init__(self,canvas,parent,coordinates=False)
		self.canvas=canvas
		self.addRateField()
		self.coordLabel=QW.QLabel("")
		self.addWidget(self.coordLabel)
		self.toggleRate(False)
		
	def toggleRate(self,state):
		self.rateField.setVisible(state)
		self.rateLabel.setVisible(state)

	def addRateField(self):
		self.rateField=QW.QLineEdit("1")
		self.rateLabel=QW.QLabel("Hz")
		self.rateField.setFixedWidth(40)
		self.rateField.setMaxLength(3)
		self.rateW=QW.QWidget()
		self.rateW.setFixedWidth(90)
		rateVBL=QW.QHBoxLayout(self.rateW)
		rateVBL.addWidget(self.rateField)
		rateVBL.addWidget(self.rateLabel)
		self.addWidget(self.rateW)
		self.addSeparator()
		
class VikFigureCanvas(FigureCanvas):
	def __init__(self,figure):
		FigureCanvas.__init__(self,figure)
		
class VikFigure(QW.QWidget):
	setLabel=pyqtSignal(str)
	def __init__(self,parent=None):
		QW.QWidget.__init__(self,parent)
		self.setWindowTitle("VikFigure")
		self.canvas=VikFigureCanvas(Figure())
		self.canvas.setFocusPolicy(Qt.ClickFocus)
		self.canvas.setFocus()
		self.toolbar=VikToolbar(self.canvas,self)
		self.layout=QW.QVBoxLayout(self)
		self.layout.addWidget(self.toolbar)
		self.layout.addWidget(self.canvas)
		self.artists=[[]]
		self.coords=[0,0]
		self.cbars=[]
		self.path=[]
		self.path_reg=[]
		self.path_test=[]
		self.path_texts=[]
		self.path_colls=[]
		self.edge_ticks_only=[]
		self.canvas.mpl_connect('motion_notify_event',self.on_mouse_moved)
		self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
		self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
		self.canvas.mpl_connect('key_press_event',self.on_key_press)
		self.canvas.mpl_connect('pick_event', self.on_pick)

		self.setLabel.connect(self.toolbar.coordLabel.setText)
		
	def setTime(self,index):
		time=self.time[index]
		if len(self.coords)==2:
			self.coords.insert(0,time)
		else:
			self.coords[0]=time
		self.setCoords()
	
	def setCoords(self):
		self.coords=[np.nan if "MaskedConstant" in str(type(c)) else c for c in self.coords]
		self.setLabel.emit(("("+",".join(["{:.4f}"]*len(self.coords))+")").format(*self.coords))
		
	def set_cmap(self,cmap):
		for art in self.artists:
			if hasattr(art[0],"set_cmap"):
				self.addColorBar(cmap,art[0].axes)
				art[0].set_cmap(cmap)
		self.canvas.draw()
		
	def reverse_cmap(self):
		self.set_cmap(reverse_cmap(self.cmap))
		
	def addColorBar(self,cmap,cbar_ax):
		norm,sm=mappable(0,1,cmap)
		if hasattr(self,"cbar"):
			# rect=cbar_ax.bbox._bbox.bounds
			self.cbar.remove()
			# cbar_ax=self.add_axes(rect)
		self.cbars.append(self.colorbar(sm,cax=cbar_ax))
		if self.edge_ticks_only:
			self.cbars[-1].set_ticks(self.cbars[-1].get_clim())
			self.cbars[-1].set_ticklabels([str(x) for x in self.edge_ticks_only])
		else:
			cbar_ax.yaxis.set_major_locator(plt.NullLocator())
		return self.cbars[-1]
		
	@property
	def cmap(self):
		return self.cbar.cmap.name
		
	def update_path(self):
		if len(self.path_test)>0:
			index=np.argmax([np.dot(v,self.camera_dir()) for v in self.path_test])
			self.path.append(self.path_test[index])
			self.path_test=[]
			x2=self.path[-1]
			self.path_texts.append(self.axes[0].text3D(*x2,"{:.3f}\n{:.3f}\n{:.3f}".format(*x2),fontsize=12,color="black"))
		if len(self.path)>1:
			x1=self.path[-2]
			L12=np.linalg.norm(x2-x1)
			if len(self.path)>2:
				x0=self.path[-3]
				L01=np.linalg.norm(x1-x0)
				cosangle=np.dot(x2-x1,x0-x1)/(L01*L12)
				angle=np.arccos(cosangle)
				v2=ortogonal_component(x0-x1,(x2-x1)/L12)
				t=np.linspace(0,angle,80)
				R=max(L01,L12)/10
				r=x1[:,None]+(x2-x1)[:,None]*np.cos(t)[None,:]*R/L12+v2[:,None]*np.sin(t)[None,:]*R/np.linalg.norm(v2)
				self.path_texts.append(self.axes[0].text3D(*2*r[:,40]-x1,"{:.1f}".format(angle*180/np.pi),ha="center",va="center",fontsize=12,color="black"))
				self.path_colls.append(self.axes[0].plot3D(*r,'-',color="black")[0])
			self.path_texts.append(self.axes[0].text3D(*(x1+x2)/2,"{:.2f}".format(L12),fontsize=12,color="black"))
			self.path_colls.append(self.axes[0].plot3D(*np.array((x1,x2)).T,'k--')[0])
		else:
			while self.path_texts:
				self.path_texts.pop(-1).remove()
			while self.path_colls:
				self.path_colls.pop(-1).remove()
		
		self.canvas.draw()
		
	def on_mouse_press(self, event):
		modifiers = QW.QApplication.keyboardModifiers()
		if modifiers == Qt.ControlModifier:
			if event.button!=1:
				self.hndlContextMenu(event)
			else:
				print(f"(x,y) = {event.xdata:.2e},{event.ydata:.2e})")
			
	def on_key_press(self,event):
		if event.key == "delete":
			self.path=[]
			self.update_path()	
			
	def on_mouse_release(self,event):
		pass

	def set_fig_size(self,w,h,dpi=None,units="cm"):
		if dpi is None:
			dpi=self.dpi
		else:
			self.canvas.figure.dpi=dpi
		f={"cm":2.54,"pt":dpi}
		self.canvas.figure.set_size_inches(*(D/f.get(units,1) if D else None for D in (w,h)))
		self.canvas.draw()
		
	def camera_dir(self):
		elev=np.pi/180*self.axes[0].elev
		azim=np.pi/180*self.axes[0].azim
		return np.array((np.cos(elev)*np.cos(azim),np.cos(elev)*np.sin(azim),np.sin(elev)))
		
	def on_pick(self,event):
		pass
						
	def hndlContextMenu(self,event):
		popMenu = QW.QMenu(self)
		if hasattr(self,"last_action"):
			popMenu.addAction("Repeat",self.last_action)
		find_menu=QW.QMenu("Find",self)
		find_ext=[QW.QMenu(m,self) for m in ("max","min","relative max","relative min")]
		addXYActions(find_ext,self.extrem,event,self)
		[find_menu.addMenu(f) for f in find_ext]
		popMenu.addMenu(find_menu)

		fit_menu=QW.QMenu("Fit",self)
		fit_peak=[QW.QMenu(m,self) for m in ("Gaussian Peak","Lorentzian Peak","Vogtian Peak")]
		addXYActions(fit_peak,self.fit_peak,event,self)
		[fit_menu.addMenu(f) for f in fit_peak]
		popMenu.addMenu(fit_menu)
	# 	else:
	# 		popMenu.addAction("Reverse",self.reverse_cmap)
	# 		menu=QW.QMenu("Select",self)
	# 		addCMActions(menu,self.set_cmap,self)
	# 		popMenu.addMenu(menu)
		cursor = QG.QCursor()
		popMenu.popup(cursor.pos())
			
	def get_viewed_data(self,ax):
		im=ax.get_images()[0]
		(xmin,xmax),(ymin,ymax)=ax.get_xlim(),ax.get_ylim()

		x0,x1,y0,y1=im.get_extent()

		if ymin>ymax:
			ymin,ymax=ymax,ymin
		
		if y0>y1:
			y0,y1=y1,y0

		print("xmin",xmin,"x0",x0)
		print("xmax",xmax,"x1",x1)
		print("ymin",ymin,"y0",y0)
		print("ymax",ymax,"y1",y1)

		z=im.get_array().data.T
		Nx,Ny=z.shape
		x=np.linspace(x0,x1,Nx)
		y=np.linspace(y0,y1,Ny)

		xf=(x>xmin)*(x<xmax)
		yf=(y>ymin)*(y<ymax)

		z=z[xf].T
		z=z[yf].T
		x=x[xf]
		y=y[yf]

		return x,y,z
		
	def fit_peak(self,value,axis,type_):
		print("not implemented")
		# if "gauss" in type_.lower():
		# 	fun=al.gauss
		# elif "voigt" in type_.lower():
		# 	fun=al.voigt

	
	def extrem(self,ax,value,axis,rel,type_):
		self.last_action=fts.partial(self.extrem,ax,value,axis,rel,type_)
		x,y,z=self.get_viewed_data(ax)
		if rel:
			ch=type_.split(' ')[-1]
			ix,iy=eval(f"sig.argrel{ch}")(z,axis=axis,order=5)
			X=x[ix]
			Y=y[iy]
		else:
			inds=eval(f"np.arg{type_}")(z,axis=axis)
			if axis==0:
				X=x[inds]
				Y=y[:]
				A=np.array((np.ones_like(Y),Y,Y*Y)).T
				a,b,c=np.linalg.lstsq(A,X)[0]
				ax.plot(a+b*Y+c*Y*Y,X,'b--')
				xext=a-b**2/(4*c)
				text="{} = ({:.3f},{:.3f})".format(type_,xext,-b/(2*c))
				xp,yp=xext,-b/(2*c)
				pol="x={:.3f}{:+.3f}y{:+.3f}y^2".format(a,b,c)
				
			else:
				X=x[:]
				Y=y[inds]
				A=np.array((np.ones_like(X),X,X*X)).T
				a,b,c=np.linalg.lstsq(A,Y)[0]
				ax.plot(X,a+b*X+c*X*X,'b--')
				yext=a-b**2/(4*c)
				text="{} = ({:.3f},{:.3f})".format(type_,-b/(2*c),yext)
				xp,yp=-b/(2*c),yext
				pol="y={:.3f}{:+.3f}x{:+.3f}x^2".format(a,b,c)
			print(X,Y)
			print(text,"fit = ({:.3f},{:.3f},{:.3f})".format(a,b,c))
			ax.annotate(text+"\n$"+pol+"$",xy=(xp,yp),xytext=(xp,yp-np.sign(c)),
        		horizontalalignment='center', verticalalignment='bottom',
				color="black",fontsize=14)

		ax.plot(X,Y,'ok',markersize=2)

		self.canvas.draw()
		
		
	def on_mouse_moved(self,event):
		for ax in self.axes:
			if event.inaxes==ax:
				self.coords=[event.xdata,event.ydata]
				if hasattr(self,"time"):
					self.coords.insert(0,self.time[self.slider.value()])
				self.setCoords()
			
	def on_mouse_scroll(self,event):
		pass
		# zoom_in(1.5*event.step)
		# pass
		# margin=np.array(self.axes[0].margins())
		# margin=np.cos(margin+event.step)
		# self.axes[0].margins(*margin)
		# self.axes[0].axis("auto")
		# # self.axes[0].view_init(elev=self.axes[0].elev+=event.step), azim=45*np.pi/180)
		# self.canvas.draw()
	
	def set_tlim(self,start,end=None):
		if not end is None:
			if (end-start)*(self.time[-1]-self.time[0])<0:
				self.time=self.time[::-1]
			end=np.argmin(np.abs(self.time-end))
		start=np.argmin(np.abs(self.time-start))
		self.slider.setRange(start,end)
	
	def update_clim(self,ax,clim=None):
		index=self.axes.index(ax)
		if clim is None:
			clim=ax.get_ylim()
		else:
			ax.set_ylim(clim)
		for art in self.artists:
			for a in art:
				if hasattr(a,"axes"):
					if not a.axes is ax and hasattr(a,"set_clim"):
						a.set_clim(clim)
		self.canvas.draw()

	def set_clabel(self,lab):
		if self.axes:
			self.axes[1].set_ylabel(lab)

	def add_slider(self,t):
		self.slider=VikSlider(parent=self)
		self.slider.setRange(0,len(t)-1)
		self.layout.insertWidget(1,self.slider)
		self.slider.valueChanged.connect(self.setArtist)
		self.slider.valueChanged.connect(self.setTime)
		self.toolbar.rateField.textChanged.connect(self.slider.setRate)
		self.toolbar.toggleRate(True)
		
	def setArtist(self,index=0):
		if hasattr(self,"slider"):
			index=self.slider.value()
		for i,art in enumerate(self.artists):
			for j in range(len(self.artists[i])):
				if hasattr(self.artists[i][j],"collections"):
					[c.set_visible(i==index) for c in self.artists[i][j].collections]
				else:
					self.artists[i][j].set_visible(i==index)
		self.canvas.draw()
		
	def __getattr__(self,attr):
		if attr=="text":
			return self.axes[0].text
		if hasattr(Figure,attr) or hasattr(Axes,attr):
			if hasattr(Figure,attr):
				return getattr(self.canvas.figure,attr)					
			elif not self.axes:
				ax=self.add_axes([0.15,0.15,0.7,0.7])
			if attr in ["plot","errorbar","scatter","imshow","pcolormesh","contour","contourf"]:
				return fts.partial(self.fun_wrap,attr)
			return getattr(self.axes[0],attr)
		return QW.QWidget.__getattr__(self,attr)

	def hide_3daxes(self,ax=None):
		if ax is None:
			ax=self.axes[0]
		ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
		ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
                        
		ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
		ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
		ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                    
		ax.set_xticks([])                               
		ax.set_yticks([])                               
		ax.set_zticks([])

	def draw_boxes(self,x,y,z,a,b,c,color=(0,0,0),alpha=1.0,center=True,type_="wireframe",**kw):
		if not center:
			x-=a/2
			y-=b/2
			z-=c/2
		u=np.linspace(0.25*np.pi,2.25*np.pi,5)
		vxz=np.arctan(np.sqrt(2))
		v=np.r_[vxz,np.pi-vxz]
		u,v=np.meshgrid(u,v,indexing="ij")
		X = np.sqrt(3)*np.cos(u)*np.sin(v)
		Y = np.sqrt(3)*np.sin(u)*np.sin(v)
		Z = np.ones_like(u)*np.cos(v)*np.sqrt(3)
		if isinstance(color,str):
			color=colors.to_rgb(color)
		N=kw.pop("N",20)
		cond = lambda t: isinstance(t,Iterable) and not (isinstance(t,tuple) and isinstance(t[0],Number))	
		L=max([len(t) for t in (x,y,z,a,b,c,alpha) if isinstance(t,Iterable)]+[1])
		args = (t if cond(t) else [t]*L for t in (x,y,z,a,b,c,color,alpha))
		ax=kw.pop("axis",self.axes[0] if self.axes else self.add_axes([0.05,0.05,0.9,0.9],projection="3d"))
		for xi,yi,zi,ai,bi,ci,coli,ali in zip(*args):
			getattr(ax,"plot_"+type_)(ai*X/2+xi,bi*Y/2+yi,ci*Z/2+zi,color=coli,alpha=ali,**kw)
		set_axes_equal(ax)
		
	def draw_spheres(self,x,y,z,r,color=(0,0,0),alpha=1.0,type_="wireframe",**kw):
		if isinstance(color,str):
			color=colors.to_rgb(color)
		N=kw.pop("N",20)
		u, v = np.mgrid[0:2*np.pi:N*1j, 0:np.pi:N*1j]
		X = r*np.cos(u)*np.sin(v)
		Y = r*np.sin(u)*np.sin(v)
		Z = r*np.ones_like(u)*np.cos(v)
		
		cond = lambda t: isinstance(t,Iterable) and not (isinstance(t,tuple) and isinstance(t[0],Number))	
		L=max([len(t) for t in (x,y,z,r) if isinstance(t,Iterable)]+[1])
		args = (t if cond(t) else [t]*L for t in (x,y,z,r,color,alpha))
		if not self.axes:
			self.add_axes([0.05,0.05,0.9,0.9],projection="3d")
		for xi,yi,zi,ri,coli,ali in zip(*args):
			getattr(self.axes[0],"plot_"+type_)(X+xi,Y+yi,Z+zi,color=coli,alpha=ali,**kw)
		set_axes_equal(self.axes[0])
		
	def plot_many(self,*arg,**kw):
		color=kw.pop("color",'black')
		scale=kw.pop("scale",1)
		if isinstance(arg[-1],str):
			*arg,kw["fmt"]=arg
		for dx,x,y in zip(*arg):
			x0=x+dx*scale
			self.plot(x0,y,color=color,**kw)

	def clabel(self,*contours,**kw):
		for index,c in enumerate(contours):
			ax=kw.get("axis",self.axes[0])
			self.artists[index].extend(ax.clabel(c,**kw))
		self.setArtist()

	def fun_wrap(self,attr,*arg,**kw):
		aic=kw.pop("alpha_in_cm",None)
		if self.artists[0] and not "cmap" in kw:
			kw["cmap"]=plt.cm.Greys
		elif aic is not None:
			com=kw["cmap"]
			if isinstance(kw["cmap"],str):
				com=plt.get_cmap(kw["cmap"])
			new_com=com(np.arange(com.N))
			new_com[:,-1] = np.linspace(0, 1, com.N)
			kw["cmap"]=ListedColormap(new_com)
		rx=kw.pop("rx",False)
		ry=kw.pop("ry",False)
		no_cbar=kw.pop("no_cbar",False)
		axis=kw.pop("axis",self.axes[0])
		self.edge_ticks_only=kw.pop("eto",[])
		if "time" in kw:
			had_time=hasattr(self,"time")
			self.time=kw.pop("time")
			if not had_time:
				self.add_slider(self.time)
			val=kw.pop("t0",None)
			if val is None or val<self.time[0]:
				value=0
			elif val>=self.time[-1]:
				value=len(self.time)-1
			else:
				value=np.argmin(np.abs(self.time-val))
			self.slider.setValue(value)
		if hasattr(self,"time"):
			new_args=[a if hasattr(a,"ndim") and a.shape==arg[-1].shape else [a]*len(self.time) for a in arg]
		else:
			new_args=[[a] for a in arg]

			self.toolbar.toggleRate(False)
		chrs="xy"
		if (attr=="plot" or attr=="errorbar") and "cmap" in kw:
			kw.pop("cmap")
		else:	
			# if isinstance(arg[-1],str):
			# 	n=-2
			# else:
			# 	n=-1
			# str_args=arg[n:]
			self.data=arg[-1]
			self.coordinates=arg[:-1]
			if not self.coordinates:
				self.coordinates=np.mgrid[0:self.data.shape[0],0:self.data.shape[1]]
			if "vmin" in kw:
				vmin=kw["vmin"]
			else:
				vmin=np.nanmin(arg[-1])
			if "vmax" in kw:
				vmax=kw["vmax"]
			else:
				vmax=np.nanmax(arg[-1])
			if "cmap" in kw and not no_cbar:
				nav_ax=self.addBorderAxis("right",0.3,padding=0.05,axis=axis)
				nav_ax.callbacks.connect('ylim_changed', self.update_clim)
				nav_ax.set_ylim(vmin,vmax)
				if self.edge_ticks_only:
					nav_ax.yaxis.set_major_locator(plt.NullLocator())
				cbar_ax=self.addBorderAxis("right",0.3,padding=0.05,facecolor='none',axis=axis)
				if len(self.artists[0])==len(self.axes)//3-1:
					self.addColorBar(kw["cmap"],cbar_ax)
				chrs+="c"
		for index,cargs in enumerate(zip(*new_args)):
			special_args=[c for c in cargs if isinstance(c,list) or isinstance(c,tuple) or hasattr(c,"ndim")]
			extra=list(cargs[len(special_args):])
			if attr=="imshow":
				kw.setdefault("aspect","auto")
				kw.setdefault("origin","lower")
				kw.setdefault("vmin",vmin)
				kw.setdefault("vmax",vmax)
				special_args=[special_args[-1].T]
				if len(cargs)==3:
					ext=lambda c,g: c.min() if g>0 else c.max()
					b1=lambda c,g: ext(c,g)-g
					b2=lambda c,g: ext(c,-g)+g
					left,bottom=(b1(c,np.mean(np.diff(c,axis=i%c.ndim))/2) for i,c in enumerate(cargs[:2]))
					right,top=(b2(c,np.mean(np.diff(c,axis=i%c.ndim))/2) for i,c in enumerate(cargs[:2]))
					
					# if left>right:
					# 	rx=not rx
					# if bottom>top:
					# 	ry=not ry
					kw["extent"]=(left,right,bottom,top)
			if (special_args[:-1] and all((isinstance(a,list) and np.array(a).ndim==1) or a.ndim==1 for a in special_args[:-1])):
				special_args[:-1]=np.meshgrid(*special_args[:-1],indexing="ij")
			artist=getattr(axis,attr)(*(special_args+extra),**kw)
			if not rx==axis.xaxis_inverted():
				axis.invert_xaxis()
			if not ry==axis.yaxis_inverted():
				axis.invert_yaxis()
			if isinstance(artist,Iterable):
				artist=artist[0]
			if hasattr(artist,"collections"):
				[c.set_visible(False) for c in artist.collections]
			else:
				artist.set_visible(False)
			if index>=len(self.artists):
				self.artists.append([])
			self.artists[index].append(artist)
		
		for ch,a in zip(chrs,arg):
			if hasattr(a,"label"):
				axis_index=self.axes.index(axis)
				getattr(self.axes[axis_index+(ch=="c")],"set_{}label".format(ch.replace("c","y")))(a.label)
		self.setArtist()
		return [a[-1] for a in self.artists]

	def addBorderAxis(self,pos,size,padding=0,**kw):
		axis=kw.pop("axis",self.axes[0])
		axis=kw.pop("axis",self.axes[0])
		rx,ry,rw,rh=axis.bbox._bbox.bounds
		if pos=="right":
			ax=self.add_axes([rx+rw+rx*padding,ry,rx*size,rh],**kw)
			ax.yaxis.set_ticks_position(pos)
			ax.yaxis.set_label_position(pos)
			ax.xaxis.set_major_locator(plt.NullLocator())
			
		if pos=="top":
			ax=self.add_axes([rx,ry+rh+ry*padding,rw,ry*size],**kw)
			ax.xaxis.set_ticks_position(pos)
			ax.xaxis.set_label_position(pos)
			ax.yaxis.set_major_locator(plt.NullLocator())
		return ax

class BaseViewer(QW.QMainWindow):
	'''Viewer base class'''
	def __init__(self):
		self._init_app()
		QW.QMainWindow.__init__(self)
		self._init_widgets()
		self._init_menus()
		self._init_actions()

	def _init_widgets(self):
		pass
	
	def _init_menus(self):
		pass
		
	def _init_actions(self):
		pass
		
	def _init_actions(self):
		self.setAttribute(Qt.WA_DeleteOnClose,False)
		
	def fileQuit(self):
		self.close()

	def closeEvent(self, ce):
		self.fileQuit()

	def about(self):
		QW.QMessageBox.about(self, "About",self.__doc__)
	
	def run(self):
		self.show()
		self.app.exec_()
		
	def _init_app(self):
		self.app = QW.QApplication.instance()
		if not self.app:
			self.app=QW.QApplication([])
		
class VikViewer(BaseViewer):
	"""Viewer to look through 2D and 3D data sets."""
	def __init__(self):
		BaseViewer.__init__(self)
		self._init_frame("VikViewer",1000,800,100,100)
		
	def _init_frame(self,title,w,h,w0,h0):
		self.setWindowTitle(title)
		rect = QW.QDesktopWidget().availableGeometry()
		x = (rect.width() - w) // 2
		y = (rect.height() - h) // 2
		self.setMinimumSize(w0, h0)
		self.setGeometry(x,y, w, h)
		
	def _init_widgets(self):
		mainSplitter=QW.QSplitter(Qt.Horizontal,self)
		viewGroup=QW.QGroupBox("Views")
		self.viewView=QW.QListView(self)
		
		self.viewModel=QG.QStandardItemModel()
		self.viewView.setModel(self.viewModel)
		self.viewView.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
		self.viewView.selectionModel().currentRowChanged.connect(self.setView)
		
		viewLayout=QW.QVBoxLayout(viewGroup)
		viewLayout.addWidget(self.viewView)
		self.views=QW.QStackedWidget(self)
		
		mainSplitter.addWidget(viewGroup)
		mainSplitter.addWidget(self.views)
		
		self.setCentralWidget(mainSplitter)

		viewGroup.setMaximumWidth(400)#self.viewView.sizeHintForColumn(0))
		
	def _init_menus(self):
		self.fileMenu = QW.QMenu('&File', self)
		self.menuBar().addMenu(self.fileMenu)
		
		self.viewMenu = QW.QMenu('&View', self)
		self.CMMenu=QW.QMenu("Set colormaps for selected views")
		self.viewMenu.addMenu(self.CMMenu)
		self.menuBar().addMenu(self.viewMenu)

		self.helpMenu = QW.QMenu('&Help', self)
		self.menuBar().addMenu(self.helpMenu)
		
	def _init_actions(self):
		self.fileMenu.addAction('&Quit', self.fileQuit,Qt.CTRL + Qt.Key_Q)
		self.fileMenu.addAction('&Save selected views', self.saveFigures,Qt.CTRL + Qt.Key_S)
		
		
		addCMActions(self.CMMenu,self.setCMS,self)
		self.viewMenu.addAction('Reverse color&maps for selected views',self.reverseAllCMS,Qt.ALT + Qt.Key_M)
		self.viewMenu.addAction('Set s&ize of selected figures', self.setFigureSizes,Qt.CTRL + Qt.Key_I)
		self.helpMenu.addAction('&About', self.about)
		
	def saveFigures(self):
		types = "PNG (*.png);;PDF (*.pdf);;JPG (*.jpg)"#;;MP4 (*.mp4)"
		fullname,ext=QW.QFileDialog.getSaveFileName(self,"Save Figures ($name and $i can be used)","image_$i.png",filter=types)
		for index,row in enumerate(self.selectedRows()):
			view=self.views.widget(row)
			fig=view.canvas.figure
			name=self.viewModel.index(row, 0 ).data(Qt.DisplayRole )
			for ch in ":/":
				name=name.replace(ch,"_")
			path=fullname.replace("$name",name).replace("$i",str(index))
			# if ".mp4" in fullname:
			# 	delay=1000/view.slider.rate
			# 	print(fullname)
			# 	anim.ArtistAnimation(fig,view.artists,interval=delay,blit=True).save(path)
			# else:
			
			fig.savefig(path,bbox_inches='tight')

	def setFigureSizes(self):
		data=gl.getInput(dict(DPI=100.0,Width=800.0,Height=600.0,Units=["pt","cm","inches",]),title="Set size")
		for row in self.selectedRows():
			widget=self.views.widget(row)
			widget.set_fig_size(data["Width"],data["Height"],data["DPI"],data["Units"])
		
	def setCMS(self,cmap):
		for row in self.selectedRows():
			self.views.widget(row).set_cmap(cmap)
			
	def reverseAllCMS(self):
		for row in self.selectedRows():
			if hasattr(self.views.widget(row),"cmap"):
				self.views.widget(row).set_cmap(reverse_cmap(self.views.widget(row).cmap))
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Delete:
			self.deleteView()
		if event.key() == Qt.Key_Space or event.key() == Qt.Key_P:
			self.views.currentWidget().slider.playPause()
		return BaseViewer.keyPressEvent(self,event)
		
	def selectedRows(self):
		return [index.row() for index in self.viewView.selectionModel().selectedIndexes()]
		
	def deleteView(self):
		rows=self.selectedRows()
		rows.sort()
		for row in rows[::-1]:
			self.viewModel.removeRow(row)
			self.views.removeWidget(self.views.widget(row))
		self.viewView.setCurrentIndex(self.viewModel.createIndex(row-(row==self.views.count()),0))
	
	def setView(self,index1,index2):
		self.views.setCurrentIndex(index1.row())
		
	def addFigure(self,name,fig):
		self.views.addWidget(fig)
		self.views.setCurrentIndex(self.views.count()-1)
		self.viewModel.appendRow(QG.QStandardItem(name))
	

if __name__=="__main__":
	#Example usage
	vv=VikViewer()
	fig_scatter=VikFigure()
	fig_scatter.scatter(*np.random.normal(*np.random.random((2,2,1)),(2,1000)))
	vv.addFigure("scatter",fig_scatter)

	fig_imshow2d=VikFigure()
	x=np.linspace(0,1,50)
	y=np.linspace(3,5,50)
	t=np.linspace(0,1,50)
	fig_imshow2d.imshow(x,y,np.random.random((50,50)),cmap="jet")
	vv.addFigure("imshow 2d",fig_imshow2d)

	fig_imshow3d=VikFigure()
	fig_imshow3d.imshow(x,y,np.random.random((50,50,50)),time=t,cmap="brg")
	vv.addFigure("imshow 3d",fig_imshow3d)

	fig_plot3d=VikFigure()
	y=np.cos(10*x[None,:])*np.sin(5*t[:,None])
	fig_plot3d.plot(x,y,time=t,color="k")
	vv.addFigure("plot 3d",fig_plot3d)

	vv.show()