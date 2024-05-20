library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggbeeswarm)
library(extrafont)
extrafont::loadfonts(quiet = TRUE)
library(gridExtra)
library(lme4)
library(emmeans)
library(nortest)
library(ggsignif)

basic_plot<- theme_bw()+
  theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),axis.title= element_text(size=20), axis.line.x = element_blank(), axis.line.y = element_blank(),
        axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18), legend.text=element_text(size=20), legend.title = element_text(size = 20),
        panel.border = element_rect(fill=NA, colour = "black", linewidth=1)) 

#source('/home/eebrandt/projects/standard_R_stuff.R')
data <- '/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/'
analysis <- '/home/eebrandt/projects/UChicago/fly_walking/sandpaper/analysis/R_raw/'
setwd('/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/')

plotcols <- c("#dc9615ff", "#3f47a4ff", "#256f33ff", "#e02125ff", "#d6e312ff")

alldata <- read.csv("combined.csv")
alldata <- subset(alldata, alldata$fly!="CA48")
names(alldata)[1] <- "ID"

metadata <- read.csv("/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/tracking/metadata.csv")
metadata <- subset(metadata, metadata$individual!="CA48")
metadata[c('ID', 'treatment', "iteration", "vid", "vid2")] <- str_split_fixed(metadata$video, '_', 5)
metadata$fulltrial <- paste(metadata$iteration, metadata$vid, metadata$vid2, sep = "_")
metadata <- metadata %>% relocate(fulltrial, .after=individual)

metadata$fulltrial = substr(metadata$fulltrial,1,nchar(metadata$fulltrial)-4)
metadata$total_ID = paste(metadata$individual, metadata$treatment, metadata$fulltrial, sep = "_")
metadata <- metadata %>% relocate(total_ID, .before=species)
colnames(metadata)[13] <- "bl_measured"

metadata$dtgassed <- paste(metadata$date_gassed, metadata$time_gassed, sep = ":")
metadata$dtgassed = as.POSIXct(metadata$dtgassed, format='%m/%d/%y:%H:%M')
metadata$dtmeasured = paste(metadata$date_recorded, metadata$time_recorded, sep = ":")
metadata$dtmeasured = as.POSIXct(metadata$dtmeasured, format='%m/%d/%y:%H:%M')

metadata$timediff =  metadata$dtmeasured - metadata$dtgassed


infodata <- metadata[c("total_ID", "ID", "iteration", "video", "treatment", "vid", "dtgassed", "dtmeasured", "timediff", "bl_measured")]
infodata <- subset(infodata, infodata$ID != "CA43")
infodata <- subset(infodata, infodata$ID != "CA36")
infodata <- subset(infodata, infodata$ID != "CA38")

alldata$fulltrial <- paste(alldata$iteration, alldata$trial, sep = "_")
alldata <- alldata %>% relocate(fulltrial, .after=treatment)
alldata$total_ID = paste(alldata$ID, alldata$treatment, alldata$fulltrial, sep = "_")
alldata <- alldata %>% relocate(total_ID, .after=ID)
colnames(alldata)[7] <- "bl_calculated"


finaldata <- merge(alldata, infodata)
finaldata$treatment <- factor(finaldata$treatment, levels=c('glass', 'g150', 'g100', 'g60', 'g24'))

ampdata <- gather(finaldata, "leg", "amplitude", c("L1_step_length", "L2_step_length", "L3_step_length", "R1_step_length", "R2_step_length", "R3_step_length"), -c(ID, iteration))
ampdata <- ampdata %>% relocate(bl_measured, .after=bl_calculated)
ampdata$side <- substr(ampdata$leg,1,1)
ampdata$legnum <- substr(ampdata$leg,2,2)

#ampdata = subset(ampdata, select = -c(9:92) )

amp <- ggplot(data = ampdata, aes(x= forcats::fct_rev(treatment), y = amplitude, fill = treatment))+
  geom_violin(trim=FALSE)+
  geom_boxplot(width = .2, fill = "grey")+
  scale_y_continuous(name = "Step Amplitude (mm)", limits = c(0,3.75))+
  scale_x_discrete(name = NULL)+
  scale_fill_manual(values = plotcols)+
  basic_plot+
  theme(legend.position = "none")+
  theme(text=element_text(family="Georgia"))+
  coord_flip()

amp


perdata <- gather(finaldata, "leg", "period", c("L1_period", "L2_period", "L3_period", "R1_period", "R2_period", "R3_period"), -c(ID, iteration))
perdata <- perdata %>% relocate(bl_measured, .after=bl_calculated)
perdata$side <- substr(perdata$leg,1,1)
perdata$legnum <- substr(perdata$leg,2,2)

#perdata = subset(perdata, select = -c(9:92) )

pdf(file = paste(analysis, 'amp+per.pdf'), height = 6.1, width = 9)
grid.arrange(amp, per, ncol = 1)
dev.off()




#Now we deal with COM speed

speed_all <- read.csv("speed.csv")
speed_all <- merge(speed_all, infodata)
speed_all$treatment <- factor(speed_all$treatment, levels=c('glass', 'g150', 'g100', 'g60', 'g24'))


speed_avg <- speed_all%>% 
  group_by(total_ID) %>% 
  summarise(treatment = first(treatment), ID = first(ID), iteration = first(iteration), video = first(video), avg_speed = mean(COM_speed), sd_speed = sd(COM_speed))


COM_speed <- ggplot(data = speed_all, aes(x= treatment, y = COM_speed, fill = treatment))+
        geom_violin(trim=FALSE)+
        geom_boxplot(width = .1, fill = "grey")+
        scale_y_continuous(name = "Walking Speed (bl/s)", limits = c(0, 40))+
        scale_x_discrete(name = NULL)+
        scale_fill_manual(values = plotcols)+
        basic_plot+
        theme(legend.position = "none")+
        theme(text=element_text(family="Georgia"))

COM_speed
pdf(file = paste(analysis, 'COM_speed.pdf'), height = 4, width = 5.6)
COM_speed
dev.off()




#stats time!

#first for amplitude
hist(ampdata$amplitude)
shapiro.test(ampdata$amplitude)
qqnorm(ampdata$amplitude)

bartlett.test(ampdata$amplitude, perdata$treatment)
boxplot(ampdata$amplitude~treatment, data = ampdata)


#check if treatment has an effect
TreatCheck <-aov(amplitude~treatment, data = ampdata)
#yes, has an effect
summary(TreatCheck)

#check to see if ID has an effect
IDcheck <-aov(amplitude~ID, data = ampdata)
#yes, has an effect
summary(IDcheck)

#check if order makes a difference
ItCheck <- aov(amplitude~iteration, data = ampdata)
summary(ItCheck)

#check if day/time of measurement has an effect
TimCheck <- aov(amplitude~dtmeasured, data = ampdata)
summary(TimCheck)

#check if video number has an effect
VidCheck <- aov(amplitude~vid, data = ampdata)
summary(VidCheck)

#first model, this is our best model
AmpSaturated <-lmer(amplitude~treatment + (1|ID) + (1|dtmeasured), data = ampdata, REML = FALSE)

AmpDifferent <- lmer(amplitude~treatment + (1|ID) + (1|iteration), data = ampdata, REML = FALSE)
AmpSmaller <-lmer(amplitude~treatment+(1|ID), data = ampdata, REML = FALSE)
AIC(AmpSaturated, AmpDifferent, AmpSmaller)

emmeans(AmpSaturated, pairwise ~ treatment, type = "tukey")

#then for period
hist(perdata$period)
hist(sqrt(perdata$period))
shapiro.test(sqrt(perdata$period))

qqnorm(sqrt(perdata$period))


PerSaturated <-lmer(sqrt(period)~treatment+(1|ID)+(1|dtmeasured), data = perdata, REML = FALSE)
PerSmaller <- lmer(sqrt(period)~treatment + (1|timediff), data = perdata, REML = FALSE)
PerSimple <- aov(sqrt(period)~treatment, data = perdata)
AIC(PerSaturated, PerSmaller, PerSimple)
emmeans(PerSaturated, pairwise ~ treatment, type = "tukey")
emmeans(PerSmaller, pairwise ~ treatment, type = "tukey")
emmeans(PerSimple, pairwise ~ treatment, type = "tukey")

per <- ggplot(data = perdata, aes(x= forcats::fct_rev(treatment), y = period, fill = treatment))+
  geom_violin(trim=FALSE)+
  geom_boxplot(width = .2, fill = "grey")+
  scale_y_continuous(name = "Period (s)", limits = c(0, 0.20))+
  scale_x_discrete(name = NULL)+
  scale_fill_manual(values = plotcols)+
  basic_plot+
  theme(legend.position = "none")+
  theme(text=element_text(family="Georgia"))+
  coord_flip()
  #geom_signif(comparisons = list(c("glass", "g24")), map_signif_level = function(p) sprintf("p < 0.001", p), y_position = 0.18)+
  #geom_signif(comparisons = list(c("glass", "g150")), map_signif_level = function(p) sprintf("p < 0.001", p), y_position = 0.19)+
  #geom_signif(comparisons = list(c("glass", "g60")), map_signif_level = function(p) sprintf("p = 0.01", p), y_position = 0.19)+
  #geom_signif(comparisons = list(c("g150", "g24")), map_signif_level = function(p) sprintf("p < 0.01", p), y_position = 0.19)+
  #geom_signif(comparisons = list(c("g100", "g24")), map_signif_level = function(p) sprintf("p < 0.01", p), y_position = 0.19)
  
per

#now for speed
hist(speed_all$COM_speed)
ad.test(speed_all$COM_speed)
qqnorm(sqrt(speed_all$COM_speed))
ad.test(sqrt(speed_all$COM_speed))
hist(sqrt(speed_all$COM_speed))

#check if treatment has an effect
SpTreatCheck <-aov(sqrt(COM_speed)~treatment, data = speed_all)
#yes, has an effect
summary(SpTreatCheck)

#check to see if ID has an effect
SpIDcheck <-aov(sqrt(COM_speed)~treatment, data = speed_all)
#yes, has an effect
summary(SpIDcheck)

#check if order makes a difference
SpItcheck <-aov(sqrt(COM_speed)~iteration, data = speed_all)
summary(SpItcheck)

#check if day/time of measurement has an effect
SpTimCheck <- aov(sqrt(COM_speed)~dtmeasured, data = speed_all)
summary(SpTimCheck)

#check if video number has an effect
SpVidCheck <- aov(sqrt(COM_speed)~vid, data = speed_all)
summary(SpVidCheck)

SpSaturated <-lmer(sqrt(COM_speed)~treatment + (1|ID) + (1|dtmeasured) + (1|iteration), data = speed_all, REML = FALSE)
SpSmaller <- lmer(sqrt(COM_speed)~treatment + (1|dtmeasured), data = speed_all, REML = FALSE)
SpSimple <-lmer(sqrt(COM_speed)~treatment+(1|ID), data = speed_all)

AIC(SpSaturated, SpSmaller, SpSimple)

emmeans(SpSaturated, pairwise ~ treatment, type = "tukey")



# now for follow the leader
ftl <- read.csv("leaderdata.csv")
ftl_total <- merge(ftl, infodata)
ftl_total$treatment <- factor(ftl_total$treatment, levels=c('glass', 'g150', 'g100', 'g60', 'g24'))
ftl_total <- subset(ftl_total, ftl_total$ID != "CA43")
#ftl_total <- subset(ftl_total, ftl_total$ID != "CA62")


ftl_plot <- ggplot(ftl_total, aes(x = treatment, y = phidiff, fill = treatment))+
  geom_violin(trim=FALSE)+
  geom_boxplot(width = .1, fill = "grey")+
  scale_y_continuous(name=expression("AEP"[2]*"- PEP"[1]*" (mm)"), limits = c(.25, 1.6))+
  scale_x_discrete(name = NULL)+
  scale_fill_manual(values = plotcols)+
  basic_plot+
  theme(legend.position = "none")+
  theme(text=element_text(family="Georgia"))

ftl_plot

pdf(file = paste(analysis, 'ftl.pdf'), height = 4, width = 5.7)
ftl_plot
dev.off()


ftl_plot

summary(aov(ftl_total$phidiff~ftl_total$treatment))
summary(aov(ftl_total$phidiff~ftl_total$ID))

hist(ftl_total$phidiff)
ad.test(ftl_total$phidiff)
qqnorm(ftl_total$phidiff)

hist(sqrt(ftl_total$phidiff))
qqnorm(sqrt(ftl_total$phidiff))

ftl_saturated <- SpSaturated <-lmer(sqrt(phidiff)~treatment + (1|ID) + (1|timediff), data = ftl_total, REML = FALSE)
ftl_smaller <-lmer(sqrt(phidiff)~treatment + (1|ID), data = ftl_total, REML = FALSE)
ftl_simple <- aov(sqrt(phidiff)~treatment, data = ftl_total)
AIC(ftl_saturated, ftl_smaller, ftl_simple)  
emmeans(ftl_smaller, pairwise ~ treatment, type = "tukey")

ftl_ID <- ggplot(data = speed_all, aes(x = treatment, y = COM_speed, fill = treatment))+
  geom_boxplot()+
  basic_plot

ftl_ID

speed_sum <- subset(speed_all, !is.na(speed_all$COM_speed)) %>% 
  group_by(treatment) %>% 
  tally()

speed_sum

amp_num <- subset(ampdata, !is.na(ampdata$amplitude))%>%
group_by(treatment) %>% 
  tally()
amp_num

per_num <- subset(perdata, !is.na(perdata$period))%>%
  group_by(treatment) %>% 
  tally()
per_num

phidiff_num <- subset(ftl_total, !is.na(phidiff))%>%
  group_by(treatment) %>% 
  tally()
phidiff_num

speed_means <- speed_all %>%
  # Specify group indicator, column, function
  group_by(treatment) %>%
  # Calculate the mean of the "Frequency" column for each group
  summarise_at(vars(COM_speed),list(mean = mean, sd = sd))

amp_means <- subset(ampdata, !is.na(ampdata$amplitude)) %>%
  # Specify group indicator, column, function
  group_by(treatment) %>%
  # Calculate the mean of the "Frequency" column for each group
  summarise_at(vars(amplitude),list(mean = mean, sd = sd))

per_means <- subset(perdata, !is.na(perdata$period)) %>%
  # Specify group indicator, column, function
  group_by(treatment) %>%
  # Calculate the mean of the "Frequency" column for each group
  summarise_at(vars(period),list(mean = mean, sd = sd))

ftl_means <- subset(ftl_total, !is.na(ftl_total$phidiff)) %>%
  # Specify group indicator, column, function
  group_by(treatment) %>%
  # Calculate the mean of the "Frequency" column for each group
  summarise_at(vars(phidiff),list(mean = mean, sd = sd))


speed_means
amp_means
per_means
ftl_means
