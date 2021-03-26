import PyQt5.QtWidgets as QW
import PyQt5.QtGui as QG
import PyQt5.QtCore as QC
from PyQt5.QtCore import Qt
import functools as fts
import sys,os

def createFunc(title : str):
	def f(self=None):
		print(title)
	return f
	
def createClass():
	class c:
		def __init__(self=None):
			pass
	return c

def addMenuBar(self,filename : str):
	with open(filename,'r') as file:
		data=[[next((i for i,c in enumerate(line.replace("    ","\t")) if not c=="\t"),0)]+\
		line.replace("    ","\t").strip("\n\t").split(":")+[""]*(2-line.count(":")) for line in file]
		menus=[self.menuBar()]
		for depth,title,name,short in data:
			parent=self
			if name:
				objects=name.split(".")
				objects,name=objects[:-1],objects[-1]
				for obj in objects:
					if obj and not hasattr(parent,obj):
						parent.setattr(obj,createClass())
					parent=getattr(parent,obj)
				if name and not hasattr(parent,name):
					setattr(parent,name,createFunc(title))
			menus=menus[:depth+1]
			if all(t=="-" for t in title):
				menus[-1].addSeparator()
			elif name:
				menus[-1].addAction(title,getattr(parent,name),shortcut=QG.QKeySequence(short))
			else:
				menus.append(menus[-1].addMenu(title))

def shapeText(text):
	size=QW.QLabel(text).minimumSizeHint()
	return size.width()+2,size.height()-1

class InputField(QW.QWidget):
	def __init__(self,name,cls,args=(),style="",direction=0,parent=None):
		QW.QWidget.__init__(self,parent)
		self.parent=parent
		self.set_name_and_style(name,style)
		self.layout=QW.QBoxLayout(direction,self)
		if not self.boxed:
			self.layout.addWidget(QW.QLabel(self.name))
		self.field=cls(*args)
		self.layout.addWidget(self.field)
		self.setContentsMargins(5,5,5,5)

	@property
	def boxed(self):
		return "top" in self.style

	def set_name_and_style(self,name,style):
		self.name=name.replace("_","")
		self.style=style
		if len(name)-len(self.name)==2:
			self.style="top"
			if name.startswith("_"):
				if name.endswith("_"):
					self.style+="center"
				else:
					self.style+="right"
			else:
				self.style+="left"

	def get(self):
		return None

	def paintEvent(self,event):
		p=QG.QPainter()
		p.begin(self)

		w,h=shapeText(self.name)
		color=self.palette().color(QG.QPalette.Window)
		if self.boxed:
			p.setPen(Qt.gray if self.valid() else Qt.red)
			if "center" in self.style:
				text_x=(self.width()-w)//2
			elif "right" in self.style:
				text_x=self.width()-h-w
			else:
				text_x=h
			p.drawRect(h//2,h//2,self.width()-h,self.height()-h)
			if self.name is not None:
				p.fillRect(text_x,0,w,h,color)
				p.setPen(Qt.black)
				p.drawText(text_x+2,h-2,self.name)
		else:
			p.setPen(color if self.valid() else Qt.red)
			p.drawRect(h//2,h//2,self.width()-h,self.height()-h)
		p.end()

	def set(self,value):
		pass

	def valid(self):
		return True

	def validate(self):
		self.repaint()
		if isinstance(self.parent,InputField):
			self.parent.validate()

class TextField(InputField):
	def __init__(self,name="",data="",style="",direction=0,validators=[],parent=None):
		InputField.__init__(self,name,QW.QLineEdit,(data,parent),style,direction,parent)
		self.validators=validators
		self.field.textEdited.connect(self.validate)

	def get(self):
		return self.field.text()

	def set(self,data):
		self.field.setText(str(data))

	def valid(self):
		try:
			value=self.get()
			return all(validate(value) for validate in self.validators)
		except:
			return False

class TypeField(TextField):
	def __init__(self,_type,name="",data=None,style="",direction=0,validators=[],parent=None):
		new_data=str(_type() if data is None else _type(data))
		TextField.__init__(self,name,new_data,style,direction,validators,parent)
		self._type=_type

	def get(self):
		return self._type(self.field.text())

class BoolField(InputField):
	def __init__(self,name="",data=False,style="",direction=0,validators=[],parent=None):
		InputField.__init__(self,name,QW.QCheckBox,(),style,direction,parent)
		self.set(data)

	def get(self):
		return self.field.isChecked()

	def set(self,data):
		self.field.setChecked(data)

class RadioField(InputField):
	def __init__(self,name="",data=[],init=0,style="",direction=0,validators=[],parent=None):
		InputField.__init__(self,name,QW.QWidget,(parent,),style,direction,parent)
		self.types={str(d):type(d) for d in data}
		self.rbl=QW.QBoxLayout((direction+2)%4,self.field)
		for id_,val in enumerate(data):
			rb=QW.QRadioButton(str(val))
			rb.clicked.connect(fts.partial(self._clicked,id_))
			rb.setChecked(id_==init)
			self.rbl.addWidget(rb)
		self.id_=init

	def _clicked(self,id_):
		self.id_=id_

	def get(self) -> str:
		text=self.rbl.itemAt(self.id_).widget().text()
		return self.types[text](text)

	def set(self,data):
		for id_,val in enumerate(data):
			self.layout.itemAt(id_).widget().setText(val)

class ChooseField(InputField):
	def __init__(self,name="",data=[],init=0,style="",direction=0,validators=[],parent=None):
		InputField.__init__(self,name,QW.QComboBox,(parent,),style,direction,parent)
		self.types={str(d):type(d) for d in data}

		for id_,val in enumerate(data):
			self.field.addItem(str(val))

		self.field.setCurrentIndex(init)

	def get(self):
		text=self.field.currentText()
		return self.types[text](text)

	def set(self,data):
		for id_,val in enumerate(data):
			self.field.setItemText(0,str(val))

class DictField(InputField):
	def __init__(self,name="",data={},style="",direction=0,parent=None):
		InputField.__init__(self,name,QW.QWidget,(parent,),style,direction,parent)
		layout=QW.QBoxLayout((direction+2)%4,self.field)
		self.fields={}

		for k,v in data.items():
			self.fields[k]=getField(v,k,style=style,direction=direction,parent=self)
			layout.addWidget(self.fields[k])

	def get(self):
		return {name:field.get() for name,field in self.fields.items()}

	def set(self,data : dict):
		for name,value in data:
			self.fields[name].set(value)

	def valid(self):
		return all(field.valid() for name,field in self.fields.items())


interps={
	dict:DictField,
	bool:BoolField,
	}

def getField(data,name="",**kw):
	if isinstance(data,QW.QWidget):
		return data
	elif isinstance(data,range) or isinstance(data,list):
		if len(data)<5:
			return RadioField(name,data,0,**kw)
		else:
			return ChooseField(name,data,0,**kw)
	elif isinstance(data,type):
		if data in interps:
			return interps[data](name,data,**kw)
		else:
			return TypeField(data,name,**kw)
	else:
		if type(data) in interps:
			return interps[type(data)](name,data,**kw)
		else:
			return TypeField(type(data),name,data,**kw)

def showMessage(msg):
	msgBox=QW.QMessageBox()
	msgBox.setText(msg)
	msgBox.exec()

def getInput(data,title="",msg="All fields must be valid",
			 accept="OK",reject="CANCEL",modal=True):
	dialog=QW.QDialog()

	dialog.setWindowTitle(title)
	dialog.setModal(modal)

	layout=QW.QVBoxLayout(dialog)
	layout.setSizeConstraint(QW.QLayout.SetFixedSize)
	input_field=getField(data)
	layout.addWidget(input_field)

	buttons=QW.QDialogButtonBox()
	buttons.setCenterButtons(True)

	acc=QW.QPushButton(accept)
	buttons.addButton(acc,QW.QDialogButtonBox.AcceptRole)

	rej=QW.QPushButton(reject)
	buttons.addButton(rej,QW.QDialogButtonBox.RejectRole)

	buttons.accepted.connect(lambda: dialog.accept() if input_field.valid() else showMessage(msg))
	buttons.rejected.connect(dialog.reject)

	layout.addWidget(buttons)

	if dialog.exec() == QW.QDialog.Accepted:
		return input_field.get()
	

if "__main__" in __name__:
	#Testing gui lib
	app=QW.QApplication.instance()
	if not app:
		app=QW.QApplication(sys.argv)

	x=getInput(data={"Name":"your name","Age":20,"Height":30})
	print(x)
	a=QW.QMainWindow()
	addMenuBar(a,filename="test.txt")
	a.show()



	sys.exit(app.exec_())
