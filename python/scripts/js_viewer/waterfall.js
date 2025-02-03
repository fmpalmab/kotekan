_dB = (d) => 10*Math.log10(d)
_mean = (d) => _.reduce(d,(memo, num) => memo + num, 0) / d.length || 1

function waterfall(container){
    var self=this;
    this.container=container

	this.kotekan_url = location.hostname;
	this.kotekan_port= 12048

	this.num_freqs=1024;
	this.waterfall_buffer_length=1000;
	this.waterfall_buffer_max_length=2500;
	this.waterfall_buffer_display_length=300;
	this.plot_width=512;
	this.margin=[100,100];
	this.waterfall_plot_height=500;

	this.scroll_data=[];
	this.bandpass_data=[];
	this.autocal_length=128;
	this.skip_length=20;
	this.spectrum_baseline=Array(this.num_freqs).fill(0);
	this.timearr=[];
	this.ms_per_datum=25.;
	this.freq_list=[];
	this.mode="normal";

	this.cb = new imgPlotter();
	this.cb_rect;

	this.cb.min=3;
	this.cb.max=7;

	this.disp_freq=[1417.,1425.]

	this.time=new Date().getTime();

    this.jqcontainer=$("#"+this.container)
    	.css('position','relative')
    	.attr('width',this.plot_width+this.margin[0])
    	.attr('height',this.waterfall_plot_height+this.margin[1])
		.width(this.plot_width+this.margin[0])
    	.height(this.waterfall_plot_height+this.margin[1])

	this.jqcontainer.parent().width(this.plot_width+this.margin[0])

	var waterfall_plot_div=$( "<div/>").uniqueId()
				.css({
       				'position':'relative',
    				'font-size':'8pt',
    				'height':this.waterfall_plot_height,'width':this.margin[0],
    			})
    			.attr('class','axis')
				.appendTo(this.jqcontainer)

	this.yaxis_scale = d3.time.scale.utc()
			    .domain([new Date(this.time),
			    		 new Date(this.time+this.waterfall_buffer_display_length*this.ms_per_datum)])
			    .range([0, this.waterfall_plot_height]);
    this.yaxis = d3.svg.axis().ticks(this.waterfall_plot_height*this.ms_per_datum/1000/2)
				              .scale(this.yaxis_scale).orient("left").tickFormat(d3.time.format('%H:%M:%S'))
	this.yaxisplot=d3.select('#'+waterfall_plot_div[0].id).append("svg")
	    .style("position","absolute")
	    .attr("height", this.waterfall_plot_height)
		.append("g")
	    .attr("transform", "translate(" + this.margin[0] + "," + 0 + ")")
	    .call(this.yaxis)

	this.yaxisplot.append("text")
			.attr("text-anchor","middle")
			.attr("font-size",20)
			.attr("y",-this.margin[0]+35)
			.attr("x",-this.waterfall_plot_height/2)
			.attr("transform", "rotate(-90)")
			.text("Time");

    this.scroll_canvas=$( "<canvas/>")
    						.attr('width', this.num_freqs)
    						.attr('height', this.waterfall_plot_height)
    						.width(this.plot_width)
    						.height(this.waterfall_plot_height)
    						.css({position:'absolute',left:this.margin[0]})
    						.appendTo(waterfall_plot_div)
    this.scrollbuf_canvas=$( "<canvas/>")
    						.attr('width', this.num_freqs)
    						.attr('height', this.waterfall_buffer_display_length)
    						.css('display','none')
    						.appendTo(this.jqcontainer)

    var freq_div = $("<div/>").uniqueId().height(20)
    						.css({
		           				'position':'relative',
		        				'font-size':'8pt',
	    	    				'height':20,'width':this.plot_width,
	    	    				'left':this.margin[0]
	            			})
	            			.attr('class','axis')
							.appendTo(this.jqcontainer)
	this.freq_scale = d3.scale.linear().range([0,this.plot_width]).domain([1,2]);
	this.freq_axis = d3.svg.axis().scale(this.freq_scale).orient("bottom")
	this.freq_axisplot=d3.select('#'+freq_div[0].id).append("svg")
	    .style("position","absolute")
	    .style("left",-10)
	    .attr("width", this.plot_width+20)
		.append("g")
	    .attr("transform", "translate(" + 10 + "," + 0 + ")")
	    .call(this.freq_axis)
	
	this.freq_axisplot.append("text")
			.attr("text-anchor","middle")
			.attr("font-size",20)
			.attr("x",this.plot_width/2)
			.attr("y",this.margin[1]/2)
			.text("Frequency [MHz]");

}

waterfall.prototype.draw =
	function()  {
		var self=this
		if (!((this.mode === "normal") || (this.mode === "stopped"))) return
		var now = new Date().getTime();
		var dt = now - (this.time || now);
		if (dt < 50) return;
		this.time = now;
		if (this.r > 0) {return;}
		this.r=requestAnimationFrame(function(){self.dodraw(); self.r=0;});
	}


waterfall.prototype.dodraw =
	function()  {
		var scd=this.scroll_data;
		var scroll_img=[];

		var ntimes_displayed = Math.min(scd.length,this.waterfall_buffer_display_length)
	    this.scroll_canvas.attr('height', ntimes_displayed)
		var img_mean=Array.apply(null, new Array(this.num_freqs)).map(Number.prototype.valueOf,0);
		var disp_start=Math.max(0,scd.length-this.waterfall_buffer_display_length);
		for (i=0; i<this.num_freqs; i++) {
			for (j=0; j<scd.length; j++) {
				img_mean[i]+=scd[j][i];
			}
			img_mean[i]/=scd.length;
		}
	 	var c = this.scroll_canvas[0].getContext("2d")
		c.imageSmoothingEnabled = false;
		var freq_sc = this.freq_list[this.freq_list.length-1]-this.freq_list[0]
		var freq_lo = Math.floor(this.num_freqs * (this.disp_freq[0]-this.freq_list[0])/freq_sc)
		var freq_hi = Math.ceil(this.num_freqs * (this.disp_freq[1]-this.freq_list[0])/freq_sc)
		var nf = Math.round(freq_hi-freq_lo)
		if (nf <1) nf=1
		this.scroll_canvas.attr('width',nf).css({'image-rendering': 'pixelated'})
		imageData = c.createImageData(nf,this.waterfall_buffer_display_length)
		for (j=disp_start; j<scd.length; j++){
			scroll_img[j]=[]
			for (i=0; i<nf; i++){
					ii = i+freq_lo
					scroll_img[j][i]=10*Math.log10(scd[j][ii]);
					if (this.baseline_check[0].checked) scroll_img[j][i]-=10*Math.log10(this.spectrum_baseline[ii])
					this.cb.setPixel(imageData,i,j-disp_start,scroll_img[j][i])
			}
		}
		self.freq_scale.domain([self.freq_list[freq_lo],self.freq_list[freq_hi]])
		self.freq_axisplot.call(self.freq_axis)
		c.putImageData(imageData, 0, 0);
		this.yaxis_scale.domain([ new Date(this.timearr[this.timearr.length-ntimes_displayed]*1e3),
								  new Date(this.timearr[this.timearr.length-1]*1e3) ])
		this.yaxisplot.call(this.yaxis)

		this.spectrum = _.map(_.transpose(this.scroll_data),_mean)
		var plot_spectrum = this.spectrum.slice(freq_lo,freq_hi)
		this.spectrum_latest = _.map(_.transpose(this.scroll_data),_.last)
		var plot_spectrum_latest = this.spectrum_latest.slice(freq_lo,freq_hi)
		var plot_spectrum_baseline =  this.spectrum_baseline.slice(freq_lo,freq_hi)

		var f = Array.from(this.freq_list.slice(freq_lo,freq_hi))
		var spectrum_plot_data_update = {
			x: [f,f,f],
			y: this.baseline_check[0].checked ?
				[plot_spectrum_latest.map((e,i) => _dB(e / plot_spectrum_baseline[i])),
				 [],
				 plot_spectrum.map((e,i) => _dB(e / plot_spectrum_baseline[i]))] :
				[_.map(plot_spectrum_latest,_dB),
				 _.map(plot_spectrum_baseline,_dB),
				 _.map(plot_spectrum,_dB)],
		}

		var spectrum_plot_layout_update = {
			'yaxis.range':[this.cb.min,this.cb.max],
			'yaxis.visible':[this.show_spectrum_mean,this.show_spectrum_latest,this.show_spectrum_baseline]
		}
		Plotly.update(this.spectrum_plot, spectrum_plot_data_update,spectrum_plot_layout_update);

		var spectrum_excess_plot_data_update = {
			x: [f,f],
			y: [_.map(_.map(plot_spectrum_latest,(v, i) => v / plot_spectrum_baseline[i]),_dB),
				_.map(_.map(plot_spectrum,(v, i) => v / plot_spectrum_baseline[i]),_dB),
			],
		}
		Plotly.restyle(this.spectrum_excess_plot, spectrum_excess_plot_data_update);

	}

waterfall.prototype.openSocket =
	function()
	{
		this.isopen=false;
	    this.socket = new WebSocket("ws://"+location.hostname+":8539");
	    this.socket.binaryType = "arraybuffer";
	    self=this
	    this.socket.onopen = function() {
	       console.log("Connected!");
	       this.isopen = true;
	    }
	    this.socket.onmessage = function(e) {
	       if (typeof e.data == "string") {
	       	  msg=JSON.parse(e.data)
	          console.log("Text message received: " + e.data)
	          for (var key in msg)
	          {
	          	console.log(key,msg[key])
	          }
     		  self.num_freqs=msg['nfreq'];
			 this.spectrum_baseline=Array(self.num_freqs).fill(0);
			  self.scroll_canvas.attr('width', self.num_freqs)
    		  self.scrollbuf_canvas.attr('width', self.num_freqs)
	       } else {
	       	  var msgtype = new Int8Array(e.data.slice(0,1))[0]

	       	  switch (msgtype) {
	       	  	case 1: //freq list
	       	  	  self.freq_list = new Float32Array(e.data.slice(1))
				  self.freq_scale.domain([self.freq_list[0],self.freq_list[self.num_freqs-1]])
				  self.freq_axisplot.call(self.freq_axis)
				  break;
	       	  	case 2: //timestep
		       	  var timestamp = new Float64Array(e.data.slice(1,9))[0]
		          var arr = new Float32Array(e.data.slice(9));
				  if (self.mode === "normal") {
					while (self.scroll_data.length>self.waterfall_buffer_length) {self.scroll_data.shift();}
					  self.scroll_data.push(arr);
					while (self.timearr.length>self.waterfall_buffer_length) {self.timearr.shift();}
					  self.timearr.push(timestamp);
				   }
				  else if (self.mode === "bandpass") {
					self.bandpass_data.push(arr);
					var percent = self.bandpass_data.length / (self.autocal_length+self.skip_length) * 100
					self.bandpass_button.button({label:'Taking calibration: '+percent.toFixed(2)+'%'})
					if (self.bandpass_data.length >= self.autocal_length+self.skip_length) {
						self.mode = "idle"
						for (i =0; i<self.skip_length; i++) self.bandpass_data.shift()
						self.process_if_bandpass()
					}
				  }
				  else if (self.mode === "skip") {
//					var arr = new Float32Array(e.data.slice(9));
					self.scroll_data.push(arr);
					self.timearr.push(timestamp);
					if (self.scroll_data.length >= self.skip_length) {
						for (i =0; i<self.skip_length; i++) self.scroll_data.shift()
						for (i =0; i<self.skip_length; i++) self.timearr.shift()
						self.mode = "normal"
					}
				  }
				  break;
			  }
	       }
	       self.draw();
	    }
	    this.socket.onerror = function(error) {
		};

	    this.socket.onclose = function(e) {
 			console.log("Connection closed.");
// 	        this.socket = null;
	        this.isopen = false;
	    }
	}

waterfall.prototype.closeSocket =
	function()
	{
		this.socket.close()
	}

waterfall.prototype.addFreqSlider =
	function(target, range)
	{
		var width=$("#"+target).width()
	    var self=this
	    var inrange=[-100,100]
	    var scale=(inrange[1]-inrange[0])/(range[1]-range[0])
	    var slider_height=50
		var slider_text=[]
		var marg=15
		var width=$("#"+target).width()


	    wrapper=$("<div/>").uniqueId().height(slider_height).appendTo($("#"+target))
	    freqslider=$("<div/>").uniqueId().appendTo(wrapper)
	    	.css({left:marg, 'margin-top':2*marg, width:width-2*marg})

		freqslider.slider({min:inrange[0],max:inrange[1],range:true,
						values:[(self.disp_freq[0]-range[0])*scale+inrange[0],
								(self.disp_freq[1]-range[0])*scale+inrange[0]],
						slide:function(event, ui){
							self.disp_freq[0]=(ui.values[0]-inrange[0])/scale+range[0];
							self.disp_freq[1]=(ui.values[1]-inrange[0])/scale+range[0];
							slider_text[0].attr({"text":self.disp_freq[0].toFixed(2)});
							slider_text[0].attr({"x":(ui.values[0]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});
							slider_text[1].attr({"text":self.disp_freq[1].toFixed(2)});
							slider_text[1].attr({"x":(ui.values[1]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});	
							self.draw();
						}})

		var rr=Raphael($("<div style='position:relative'/>").uniqueId().appendTo(wrapper)[0].id,width, 50);
		rr.canvas.style.position="absolute";
		rr.canvas.style.zIndex="100";
		rr.setStart();
		slider_text[0]=rr.text((self.disp_freq[0]-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,
			self.disp_freq[0].toFixed(2));
		slider_text[0].attr({'font-size': 12});
		slider_text[1]=rr.text((self.disp_freq[1]-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,
			self.disp_freq[1].toFixed(2));
		slider_text[1].attr({'font-size': 12});
		rr.text(width/2,30,"Freq Range [MHz]").attr({'font-size':14});
		rr.setFinish()
	}


waterfall.prototype.addColorSlider = 
	function(target,range)
	{
		var width=$("#"+target).width()
	    var self=this
	    var inrange=[-1000,1000]
	    var scale=(inrange[1]-inrange[0])/(range[1]-range[0])
	    var slider_height=50
		var slider_text=[]
		var marg=15
		var width=$("#"+target).width()

	    wrapper=$("<div/>").uniqueId().height(slider_height).appendTo($("#"+target))
	    cbslider=$("<div/>").uniqueId().appendTo(wrapper)
	    	.css({left:marg, 'margin-top':2*marg, width:width-2*marg})

		cbslider.slider({min:inrange[0],max:inrange[1],range:true,
						values:[(self.cb.min-range[0])*scale+inrange[0],
								(self.cb.max-range[0])*scale+inrange[0]],
						slide:function(event, ui){
							self.cb.min=(ui.values[0]-inrange[0])/scale+range[0];
							self.cb.max=(ui.values[1]-inrange[0])/scale+range[0];
							slider_text[0].attr({"text":self.cb.min.toFixed(2)});
							slider_text[0].attr({"x":(ui.values[0]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});
							slider_text[1].attr({"text":self.cb.max.toFixed(2)});
							slider_text[1].attr({"x":(ui.values[1]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});	

							for (i=0; i<self.cb_tags.length; i++){
								self.cb_tags[i].attr({"text":(i/(self.cb_tags.length-1)*(self.cb.max-self.cb.min)+self.cb.min).toFixed(2)});
							}
							self.draw();
						}})
//		cbslider.width(width);


		var rr=Raphael($("<div style='position:relative'/>").uniqueId().appendTo(wrapper)[0].id,width, 50);
		rr.canvas.style.position="absolute";
		rr.canvas.style.zIndex="100";
		rr.setStart();
		slider_text[0]=rr.text((self.cb.min-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,self.cb.min.toFixed(2));
		slider_text[0].attr({'font-size': 12});
		slider_text[1]=rr.text((self.cb.max-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,self.cb.max.toFixed(2));
		slider_text[1].attr({'font-size': 12});
		rr.text(width/2,30,"Color Bar Range [dB]").attr({'font-size':14});
		rr.setFinish()
	}

waterfall.prototype.addColorBar = 
	function(target)
	{
		var cb_height=50
		var marg=15
		var width=$("#"+target).width()

	    wrapper=$("<div style='margin:0px'/>").uniqueId().height(cb_height).appendTo($("#"+target))
	    cb=$( "<div/>").uniqueId().appendTo(wrapper)

		var rr = Raphael(cb[0].id, width,cb_height);
		rr.setStart()
		this.cb_rect=rr.rect(marg,0,width-2*marg,30).attr({fill:this.cb.cb_grad});

		this.cb_tags=[]
		var ntags=5
		for (i=0; i<ntags; i++)
		{
			this.cb_tags[i]=rr.text(i/(ntags-1)*(width-2*marg),40,(i/(ntags-1)*(this.cb.max-this.cb.min)+this.cb.min).toFixed(2))
							.attr({'text-anchor':'start'});	
		}

		rr.setFinish();
	}

waterfall.prototype.addColorSelect =
	function(target,cb2use)
	{
		var self=this
		var marg=15
		var width=$("#"+target).width()

		cp=$("<select/>").appendTo($("<div/>").appendTo("#"+target).css({margin:marg}))
		for (var newcm of cb2use){
			if (newcm in this.cb.colormaps){
			    cp.append("<option>"+newcm+"</option>")}
			else {console.log(newcm+" not a known colormap!")}
		}
		cp.selectmenu({
			change: function(event, data){self.change_palette(event, data);},
			width: width-2*marg}
		)
		.data("ui-selectmenu")
		._renderItem=function(ul, item) {
			self.li = $( "<li>", {text: item.label});
			var im = $( "<span/>")
						.appendTo(self.li)
						.css({'right':5,width:'60%',position:'absolute'});

			var rr = Raphael(im[0],"100%",Math.ceil(this.button[0].clientHeight/2));
			rr.setStart();
			rr.rect(0,0,"100%","100%").attr({fill:self.cb.gradString(self.cb.colormaps[item.label])});
			rr.setFinish();

			return self.li.appendTo(ul);
		};

	}

waterfall.prototype.start = function() {
	this.openSocket();
	this.mode = "normal";
	this.scroll_data = [];
}
waterfall.prototype.stop = function() {
	this.closeSocket();
	this.mode = "stopped";
}

waterfall.prototype.addStartStop =
	function(target)
	{
		self=this
		let label= (self.mode === "stopped")? "Start" : "Stop"
		let icon = (self.mode === "stopped")? "ui-icon-play" : "ui-icon-stop"
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target)).css({margin:45})
		self.startstop_btn = $("<button/>").appendTo($("<div/>").appendTo(wrapper))
				.button({label:label,icons:{primary: icon}})
				.css({margin:"0 auto",display:"block"})
				.css({'border':'1px solid'})
				.click(function() {
					if ( self.mode === "stopped" ) {
						$( this ).button( "option", {label: "Stop", icons: {primary: "ui-icon-stop"}})
							.css({'border':'3px solid green'})
						self.start();
				    } else {
						$( this ).button( "option", {label: "Start", icons: {primary: "ui-icon-play"}})
							.css({'border':'1px solid'})
						self.stop();
					}
				});
	}

waterfall.prototype.addRecordButton = 
	function(target)
	{
		self = this
		this.recording = false
		this.fn_idx = 0
		var marg=15
		var width=$("#"+target).width()
	    wrapper=$("<div/>").uniqueId().css({'margin':marg,'width':'100%'})
	    			.height(45).width(width-2*marg).appendTo($("#"+target))

		this.record_fn=$("<input type='text'/>")
				.css({'width':'50%','float':'left', 'font-size':'16pt', 'margin-top':5})
				.val("output"+("000" + this.fn_idx).slice(-4)+".dat")
				.appendTo(wrapper)

		this.record_btn = $("<button/>").appendTo($( "<div style='margin:10px'/>").appendTo(wrapper))
				.css({'float':'right'})
				.button({label:'Record', icons: {primary: "ui-icon-disk"}})
				.click(function() {
					if (!self.recording) {
						self.socket.send(JSON.stringify({'type': 'record', 'state': true, 'file': self.record_fn.val()}))
						$ (this ).button( "option", {label: "Recording", icons: {primary: "ui-icon-bullet"}})
						self.recording = true
					}
					else {
						self.socket.send(JSON.stringify({'type': 'record', 'state': false, 'file': "null"}))
						$ (this ).button( "option", {label: "Record", icons: {primary: "ui-icon-blank"}})
						self.recording = false
						self.fn_idx = self.fn_idx+1;
						self.record_fn.val("output"+("000" + self.fn_idx).slice(-4)+".dat")
					}
				});
	}

function download(bytes, fname)  {
		let blob = new Blob(bytes, {type:"application/octet-stream"});
		let link = document.createElement('a');
		link.href = window.URL.createObjectURL(blob);
		link.download = fname;
		link.click();
		window.URL.revokeObjectURL(link.href);
	}
	

waterfall.prototype.addLocalRecordButton = 
	function(target)
	{
		self = this
		this.recording = false
		this.fn_idx = 0
		var marg=15
		var width=$("#"+target).width()
	    wrapper=$("<div/>").uniqueId().css({'margin':marg,'width':'100%'})
	    			.height(45).width(width-2*marg).appendTo($("#"+target))

		this.record_fn=$("<input type='text'/>")
				.css({'width':'50%','float':'left', 'font-size':'16pt', 'margin-top':5})
				.val("output"+("000" + this.fn_idx).slice(-4)+".dat")
				.appendTo(wrapper)

		this.record_btn = $("<button/>").appendTo($( "<div style='margin:10px'/>").appendTo(wrapper))
				.css({'float':'right'})
				.button({label:'Save Data', icons: {primary: "ui-icon-disk"}})
				.click(function() {
					var oldmode = self.mode
					self.mode = 'idle'
					file_data = [].concat(new Int32Array([self.num_freqs,self.scroll_data.length]))
								  .concat(new Float32Array([self.CCERA.alt, self.CCERA.az]))
								  .concat(new Float32Array(self.freq_list))
								  .concat(new Float64Array(self.timearr))
								  .concat(new Float32Array(self.spectrum))
								  .concat(new Float32Array(self.spectrum_baseline))			
								  .concat(self.scroll_data)
					download(file_data, self.record_fn.val());
					self.fn_idx = self.fn_idx+1;
					self.record_fn.val("output"+("000" + self.fn_idx).slice(-4)+".dat")
					self.mode = oldmode
			});
	}


waterfall.prototype.addAirspyGainControl =
	function(target,stage_url)
	{
		self=this

		let change_gain = function(type,value){
			fetch('http://'+self.kotekan_url+':'+self.kotekan_port+'/'+stage_url+'/set_config', {
				mode: 'no-cors',
			    method: 'POST',
			    headers: {
			        'Accept': 'application/json',
			        'Content-Type': 'application/json'
			    },
			    body: JSON.stringify({[type]: value})
			})
		   .then(check_adcstats)
		}
		let check_adcstats = function(){
			fetch('http://'+self.kotekan_url+':'+self.kotekan_port+'/'+stage_url+'/adcstat',{})
				.then(r => r.json().then(data => {
					adcmean.text("Mean: "+data['mean'].toFixed(2))
					adcrms.text("RMS: "+data['rms'].toFixed(2))
					adcrailfrac.text("Rail %: "+(data['railfrac']*100).toFixed(2))
	 			}))
		}

		var marg=15
	    var slider_width=50
	    var slider_height=200
	    var wrapper=$("<div'/>").uniqueId().height('100%')
					.width((this.plot_width+this.margin[0])/2-1)
					.css({'margin':'10px', 'float':'left', 'margin':'0px'})
					.appendTo($("#"+target))

	    var gainwrap=$('<div/>').uniqueId().css({width:3*(slider_width+4)}).appendTo(wrapper)
		$("<p/>").css({'font-family':'sans-serif','text-align':'center','margin':marg})
		    		.text("Gain").appendTo(gainwrap)

	    var adcwrap=$('<div/>').uniqueId().css({width:'auto'}).appendTo(wrapper)
			    .css({'font-family':'sans-serif','font-size':'10pt','text-align':'left','margin':marg})
		$("<p>").text("ADC Stats").css({'font-size':'14pt','text-align':'center'}).appendTo(adcwrap)
		var adcmean = $("<div/>").css({'position':'relative','left':'30px'})
				.text("Mean: ").appendTo(adcwrap)
		var adcrms = $("<p/>").css({'position':'relative','left':'30px'})
				.text("RMS: ").appendTo(adcwrap)
		var adcrailfrac = $("<p/>").css({'position':'relative','left':'30px'})
				.text("Rail %: ").appendTo(adcwrap)


		var lnawrap = $("<div style='float:left'/>").width(slider_width)
					.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(gainwrap)
		$("<p/>").css({'font-family':'sans-serif', 'margin':2, 'margin-bottom':15})
				.text("LNA").appendTo(lnawrap)

		var slider_gain_lna=$("<div/>").uniqueId().appendTo(lnawrap).css({'margin':'auto'})
					.slider({min:0,max:14,value:10,step:1,
						orientation: "vertical",
						slide:function(event, ui){
							lna_gain=ui.value;
							change_gain("gain_lna",lna_gain)
							slider_gain_lnat.text(ui.value);
						}
					})
		var slider_gain_lnat=$("<p/>").css({'font-family':'sans-serif','text-align':'center','margin':2})
				.text(10).appendTo(lnawrap)

		var mixwrap = $("<div style='float:left'/>").width(slider_width)
					.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(gainwrap)
		$("<p/>").css({'font-family':'sans-serif','margin':2, 'margin-bottom':15})
				.text("MIX").appendTo(mixwrap)

		var slider_gain_mix=$("<div/>").uniqueId().appendTo(mixwrap).css({'margin':'auto'})
					.slider({min:0,max:15,value:10,step:1,
						orientation: "vertical",
						slide:function(event, ui){
							mix_gain=ui.value;
							change_gain("gain_mix",mix_gain)
							slider_gain_mixt.text(ui.value);
						}
					})
		var slider_gain_mixt=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
				.text("10").appendTo(mixwrap)

		var ifwrap = $("<div style='float:left'/>").width(slider_width)
					.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(gainwrap)
		$("<p/>").css({'font-family':'sans-serif', 'margin':2, 'margin-bottom':15})
				.text("IF").appendTo(ifwrap)

		var slider_gain_if=$("<div/>").uniqueId().appendTo(ifwrap).css({'margin':'auto'})
					.slider({min:0,max:15,value:10,step:1,
						orientation: "vertical",
						slide:function(event, ui){
							if_gain=ui.value;
							change_gain("gain_if",if_gain)
							slider_gain_ift.text(ui.value);
						}
					})
		var slider_gain_ift=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
				.text("10").appendTo(ifwrap)

		fetch('http://'+self.kotekan_url+':'+self.kotekan_port+'/'+stage_url+'/get_config',{})
			.then(r => r.json().then(data => {
				slider_gain_lna.slider('value',data["lna_gain"])
				slider_gain_lnat.text(data["lna_gain"])
				slider_gain_mix.slider('value',data["mix_gain"])
				slider_gain_mixt.text(data["mix_gain"])
				slider_gain_if.slider('value',data["if_gain"])
				slider_gain_ift.text(data["if_gain"])
			}))
		check_adcstats()

	}


	waterfall.prototype.addBufferControl =
	function(target)
	{
		self=this

		var width=$("#"+target).width()
		var marg=15
	    var self=this
	    var slider_height=50
	    var wfslider

	    wrapper=$("<div style='margin:10px'/>").uniqueId().height(slider_height).width(width-2*marg).appendTo($("#"+target))

	    var bintext=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
	    		.text("Time Samples in full Buffer:").appendTo(wrapper)
				.css({'float':'left'})

	    wfslider=$("<div style='width:50%'/>").uniqueId().appendTo(wrapper)
					.slider({min:100,max:this.waterfall_buffer_max_length,value:this.waterfall_buffer_length,
						slide:function(event, ui){
							self.waterfall_buffer_length=ui.value;
							while (self.scroll_data.length > self.waterfall_buffer_length)
								self.scroll_data.shift();
							self.draw();
							bins_text.val(ui.value);
						}})
						.css({'float':'left'})

		var bins_text=$("<input type='number'/>")
				.attr({min:100,max:this.waterfall_buffer_max_length})
				.css({'width':'17%','display':'inline','font-size':'16pt', 'margin-top':5})
				.val(this.waterfall_buffer_length)
				.appendTo(wrapper)
				.change(
					function(){
						if (parseInt(this.value) < ($(this).attr("min"))) {this.value=$(this).attr("min")}
						if (parseInt(this.value) > ($(this).attr("max"))) {this.value=$(this).attr("max")}
						wfslider.slider('value',this.value)
						self.waterfall_buffer_display_length=parseInt(this.value);
						self.draw()
					}
				)
		bins_text.numeric()
		
		wfclearbtn = $("<button/>").uniqueId().appendTo(wrapper)
				.button({label:'Clear',icons:{primary: "ui-icon-close"}})
				.css({'display':'inline-block','float':'right'})
				.click(function() {
					self.scroll_data=[]
					self.timearr=[]
					self.draw()
				});
	}


waterfall.prototype.addWaterfallControl =
	function(target)
	{
		self=this

		var width=$("#"+target).width()
		var marg=15
	    var self=this
	    var slider_height=50
	    var wfslider

	    wrapper=$("<div style='margin:10px'/>").uniqueId().height(slider_height).width(width-2*marg).appendTo($("#"+target))

	    var bintext=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
	    		.text("Time Samples in Waterfall:").appendTo(wrapper)
				.css({'float':'left'})

	    wfslider=$("<div style='width:50%'/>").uniqueId().appendTo(wrapper)
					.slider({min:100,max:this.waterfall_buffer_length,value:this.waterfall_buffer_display_length,
						slide:function(event, ui){
							self.waterfall_buffer_display_length=ui.value;
							self.draw()
							bins_text.val(ui.value);
						}})
						.css({'float':'left'})

		var bins_text=$("<input type='number'/>")
				.attr({min:100,max:this.waterfall_buffer_length})
				.css({'width':'17%','display':'inline', 'font-size':'16pt', 'margin-top':5})
				.val(this.waterfall_buffer_display_length)
				.appendTo(wrapper)
				.change(
					function(){
						if (parseInt(this.value) < ($(this).attr("min"))) {this.value=$(this).attr("min")}
						if (parseInt(this.value) > ($(this).attr("max"))) {this.value=$(this).attr("max")}
						wfslider.slider('value',this.value)
						self.waterfall_buffer_display_length=parseInt(this.value);
						self.draw()
					}
				)
		bins_text.numeric()
	}

waterfall.prototype.change_palette=
	function(event, data)
	{
		this.cb.gradientScale(this.cb.colormaps[data.item.label]);
		this.cb_rect.attr({fill:this.cb.cb_grad})
		this.draw();
	}

waterfall.prototype.add_spectrum=
	function(target){
		this.freeze_baseline = false
	    wrapper=$("<div style='margin:0px'/>").uniqueId().appendTo($("#"+target))
				.height(300).width(this.plot_width+this.margin[0]/2-1)
				.css({'margin-left':this.margin[0]/2})
		spectrum_plot_data_mean     = {x: [],y: [],type: 'scatter',name:'Mean'}			
		spectrum_plot_data_latest   = {x: [],y: [],type: 'scatter',name:'Latest',mode:'markers',
											marker: {size: 3} }
		spectrum_plot_data_baseline = {x: [],y: [],type: 'scatter',name:'Baseline'}			
		var data = [spectrum_plot_data_latest, spectrum_plot_data_baseline, spectrum_plot_data_mean];
		this.show_spectrum_mean = true
		this.show_spectrum_latest = true
		this.show_spectrum_baseline = true

		this.spectrum_plot = wrapper.attr('id')

		var layout = {
			title: {text:'Spectral Power'},
			xaxis: {title: {text: 'Frequency (MHz)'},linecolor: 'black',zeroline:false},
			yaxis: {title: {text: 'Power (dB bits^2)'},linecolor: 'black',zeroline:false},
			margin: {t:30, l:50, r:10, b:40},
			legend: {xanchor:'right',x:1.0,y:0.}
		}

		Plotly.newPlot(this.spectrum_plot, data, layout, {staticPlot: true});
	}

waterfall.prototype.add_baseline_control=
	function(target){
		self=this
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
						.css({position:'relative',float:'left'})
		self.baseline_btn = $("<button/>").appendTo(wrapper)
				.button({label:'Take a Spectral Baseline',icons:{primary: "ui-icon-play"}})
				.click(function() {
						self.spectrum_baseline = _.map(_.transpose(self.scroll_data),_mean)
						self.baseline_check_wrapper.css({visibility:'visible'})
					});
	}

waterfall.prototype.add_baseline_fitter=
	function(target){
		self = this
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
				.css({position:'relative',float:'left'})
		self.baseline_btn = $("<button/>").appendTo(wrapper)
				.button({label:'Fit Polynomial Baseline',icons:{primary: "ui-icon-play"}})
				.click(function() {
						x=[]
						y=[]
						for (idx=0; idx<self.num_freqs; idx++){
							if (((self.freq_list[idx] > self.disp_freq[0]) &&
								 (self.freq_list[idx] < 1420.4-0.5)) ||
								((self.freq_list[idx] < self.disp_freq[1]) &&
								 (self.freq_list[idx] > 1420.4+0.5))
								) {
									x.push(idx)
									y.push(self.spectrum[idx] - self.spectrum_baseline[idx])
								}
						}
						poly = Polyfit(x, y)
						solver = poly.getPolynomial(4);
						var polybase = _.map(Array.from(Array(self.num_freqs).keys()),solver)
						self.spectrum_baseline = self.spectrum_baseline.map((e,i) => e+polybase[i])
						self.baseline_check_wrapper.css({visibility:'visible'})
					});
	}

waterfall.prototype.add_baseline_subtract=
	function(target){
		self=this
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
						.css({position:'relative',float:'left'})
						.css({visibility:'hidden'})

		baseline_label = $("<label>").appendTo(wrapper)
				.text("Remove Baseline from Display")
		self.baseline_check = $("<input type='checkbox'/>").appendTo(baseline_label)
				.checkboxradio({icon:false})
				.click(function(){ self.draw()})
		self.baseline_check_wrapper = wrapper
	}
	
waterfall.prototype.add_autocal_if=
	function(target,stage_url){
		self=this
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
						.css({position:'relative',float:'left'})
		self.bandpass_button = $("<button/>").appendTo(wrapper)
				.button({label:'Take 1416MHz Bandpass',icons:{primary: "ui-icon-play"}})
				.click(function() {
					fetch('http://'+self.kotekan_url+':'+self.kotekan_port+'/'+stage_url+'/set_config', {
						mode: 'no-cors',
						method: 'POST',
						headers: {
							'Accept': 'application/json',
							'Content-Type': 'application/json'
						},
						body: JSON.stringify({"freq": 1416})
					})
					.then(gather_if_bandpass())
				});

		gather_if_bandpass = function(){
				self.bandpass_data = []
				self.scroll_data = []
				self.timearr = []
				self.mode = "bandpass"
			}
		self.process_if_bandpass = function(){
				self.spectrum_baseline = _.map(_.transpose(self.bandpass_data),_mean)

				fetch('http://'+self.kotekan_url+':'+self.kotekan_port+'/'+stage_url+'/set_config', {
					mode: 'no-cors',
					method: 'POST',
					headers: {
						'Accept': 'application/json',
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({"freq": 1421})
				})
				.then(resume_data())
			};
		
		resume_data = function(){
			self.baseline_check_wrapper.css({visibility:'visible'})
			self.bandpass_button.button({label:"Autocalibrate Bandpass"})
			self.scroll_data = []
			self.mode = "skip";
		}

		var autocal_length_text=$("<input type='number'/>")
				.attr({min:16,max:4096})
				.css({'width':'17%','display':'inline','font-size':'16pt', 'margin-top':5})
				.val(this.autocal_length)
				.appendTo(wrapper)
				.change(
					function(){
						if (parseInt(this.value) < ($(this).attr("min"))) {this.value=$(this).attr("min")}
						if (parseInt(this.value) > ($(this).attr("max"))) {this.value=$(this).attr("max")}
						self.autocal_length = parseInt(this.value);
					}
				)
		autocal_length_text.numeric()


	}


waterfall.prototype.add_spectrum_excess=
	function(target){
		this.freeze_baseline = false
	    wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
				.height(200).width(this.plot_width+this.margin[0]/2)
				.css({position:'relative',float:'left',visible:'false'})
		spectrum_plot_excess_mean     = {x: [],y: [],type: 'scatter',name:'Mean'}			
		spectrum_plot_excess_latest   = {x: [],y: [],type: 'scatter',name:'Latest',mode:'markers',
											marker: {size: 3} }
		var data = [spectrum_plot_excess_latest, spectrum_plot_excess_mean];
		this.show_spectrum_excess_mean = true
		this.show_spectrum_excess_latest = true

		this.spectrum_excess_plot = wrapper.attr('id')

		var layout = {
			title: {text:'Excess Spectral Power'},
			xaxis: {title: {text: 'Frequency (MHz)'},linecolor: 'black',zeroline:false},
			yaxis: {title: {text: 'Excess Power (dB bits^2)'},linecolor: 'black',zeroline:false,range:[-1,4]},
			margin: {t:30, l:50, r:10, b:40},
			legend: {xanchor:'right',x:1.0,y:0.}
		}

		Plotly.newPlot(this.spectrum_excess_plot, data, layout, {staticPlot: true});
	}

waterfall.prototype.addCCERAPointing=
	function(target){
		self=this
		marg=15
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
						.css({position:'relative',float:'left',width:'100%'})
		self.CCERA = {"lat":null,
					  "lon":null,
					  "el":null,
					  "alt":null,
					  "az":null,
					  "ra":null,
					  "dec":null, 
					  "l":null,
					  "b":null}

		update_pointing = function() {
			fetch('http://'+self.kotekan_url+':3000/position')
				.then(r => r.json().then(data => {
					self.CCERA.lat = data.lat
					self.CCERA.lon = data.lon
					self.CCERA.el = data.el
					lat.text("Lat: "+data['lat'].toFixed(2)+" deg")
					lon.text("Lon: "+data['lon'].toFixed(2)+" deg")
					el.text("Elev: "+data['el'].toFixed(2)+"m")
					fetch('http://'+self.kotekan_url+':3000/pointing')
						.then(r => r.json().then(data => {
						    self.CCERA.alt = data.alt
							self.CCERA.az = data.az
						    alt.text("Alt: "+data['alt'].toFixed(2)+" deg")
						    az.text("Az: "+data['az'].toFixed(2)+" deg")

							ra.text("RA: "+data['ra'].toFixed(2)+" deg")
						    dec.text("Dec: "+data['dec'].toFixed(2)+" deg")
						    gl.text("Gal. Lon: "+data['gl'].toFixed(2)+" deg")
						    gb.text("Gal. Lat: "+data['gb'].toFixed(2)+" deg")
							self.update_galview(data['gl'])
						}))
			}))
		}
		update_pointing()
		setInterval(update_pointing,5000)

		var ccerawrap=$('<div/>').uniqueId().css({width:'100%'}).appendTo(wrapper)
			.css({'position':'relative','float':'left'})
			.css({'font-family':'sans-serif','font-size':'10pt'})

		$("<p>").text("Telescope Info").css({'font-size':'14pt','text-align':'center'}).appendTo(ccerawrap)
		
		var lcol = $("<div/>").css({width:"33%",float:'left'}).appendTo(ccerawrap)
		var mcol = $("<div/>").css({width:"33%",float:'left'}).appendTo(ccerawrap)
		var rcol = $("<div/>").css({width:"33%",height:"75px",position:'relative',float:'left'}).appendTo(ccerawrap)

		var alt = $("<div/>").css({width:"100%"})
				 .text("Alt: ").appendTo(lcol)
		var az = $("<div/>").css({width:"100%"})
				 .text("Az: ").appendTo(lcol)
		var lat = $("<div/>").css({width:"100%"})
				 .text("Lat: ").appendTo(lcol)
		var lon = $("<div/>").css({width:"100%"})
				 .text("Lon: ").appendTo(lcol)
		var el = $("<div/>").css({width:"100%"})
				 .text("Elev: ").appendTo(lcol)
		var ra = $("<div/>").css({width:"100%"})
				 .text("RA: ").appendTo(mcol)
		var dec = $("<div/>").css({width:"100%"})
				 .text("Dec: ").appendTo(mcol)
		var gl = $("<div/>").css({width:"100%"})
				 .text("Gal. Lon: ").appendTo(mcol)
		var gb = $("<div/>").css({width:"100%"})
				 .text("Gal. Lat: ").appendTo(mcol)
		var state =  $("<div/>").css({width:"100%",height:"100%"})
				 .css({'display':'flex','justify-content':'center','align-items':'center'})
				 .css({'white-space':'pre-line'})
				 .css({'text-align':'center','font-size':'12pt'})
				 .text("STATE").css({'border':'2px solid black','border-radius':'5px'}).appendTo(rcol)
				 

		update_state = function() {
			fetch('http://'+self.kotekan_url+':3000/state')
				.then(r => r.json().then(data => {
						if (data['state'].startsWith("on source")) {
							var l = data['state'].split(',')[1].substring(1)
							console.log(l)
							data['state'] = "On Source \n"+l
							state.css({backgroundColor:"#90EE90"})
						}
						else if (data['state'].startsWith("slewing")) {
							data['state'] = "Slewing"
							state.css({backgroundColor:"#FFDBBB"})
						}
						else 
							state.css({backgroundColor:"white"})

						if (data['expiry']) {
							time_left = (data['expiry'] - data['now']).toFixed(0)
							state.text(data['state'] + "\n Dwell:" + time_left + "sec")
						}
						else
							state.text(data['state'])	
					}))
				}
		update_state()
		setInterval(update_state,1000)

	}


waterfall.prototype.addGalView=
	function(target, imgsrc){
		self = this
		wrapper=$("<div/>").uniqueId().appendTo($("#"+target))
				.width("100%")
				.css({position:'relative',float:'left',visible:'false'})

		galimg = $("<img/>").appendTo(wrapper)
					.attr("src",imgsrc)
					.attr({width:"100%"})
					.css({"filter":"invert(100%)"})

		var w = galimg.width()
		var h = galimg.height()

		var sun_frac_loc = [0.5,1-0.309]

		//$("<div style='position:absolute'/>").uniqueId().appendTo(wrapper)
		var rr=Raphael(wrapper[0].id,w,h);
		rr.canvas.style.position="absolute";
		rr.canvas.style.zIndex="100";
		rr.setStart()
		var los = rr.path(["M",w*sun_frac_loc[0],h*sun_frac_loc[1],
							  "L",w*sun_frac_loc[0],h*sun_frac_loc[1]])
		los.attr({"arrow-end":"classic-wide-long",})
		rr.setFinish()

		self.update_galview = function(gl){
			var h = galimg.height()
			var w = galimg.width()
			var dx = w*sun_frac_loc[0] - w/2 * Math.sin(gl/360*2*Math.PI)
			var dy = h*sun_frac_loc[1] - h/2 * Math.cos(gl/360*2*Math.PI)
			los.attr({"path":["M",w*sun_frac_loc[0],h*sun_frac_loc[1],
							  "L",dx,dy],
					  "stroke-width":3,
			})
		  }
				
	}
